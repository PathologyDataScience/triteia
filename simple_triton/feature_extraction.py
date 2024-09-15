import argparse
import os
from concurrent.futures import ProcessPoolExecutor, wait
from time import sleep, time

import histomics_stream as hs
import large_image_source_tiff
import numpy as np
import os
from simple_triton.io.tfr_writer import write_record
from simple_triton.inference import Requests
from simple_triton.model import TritonModel
from simple_triton.tile_iterators import TiffPrefetch
from time import sleep, time
from tqdm import tqdm
import tensorflow as tf

from simple_triton.inference import Requests
from simple_triton.io.tfr_writer import write_record
from simple_triton.tile_iterators import TiffPrefetch


def study(
    paths,
    t=(224, 224),
    overlap=(0, 0),
    chunk=(896, 896),
    objective=20.0,
    mask_threshold=0.01,
):
    """Convenience function for generating a histomics stream study for a single
    whole-slide image.

    Parameters
    ----------
    paths : string | (string, string)
        Path to the whole-slide image and optionally a foreground mask.
    t : tuple(int, int)
        Tile height and width (pixels) at the target magnification. Default is (224, 224).
    overlap : tuple(int, int)
        Vertical and horizontal tile overlap (pixels). Default value is (0, 0).
    chunk : (int, int)
        Size of region for grouping tiles (pixels) at the target magnification. Grouping
        tiles into a single read improves inference throughput. Default value is (896, 896).
        A value of `None` will leave tiles to be read individually.
    objective : float
        Objective magnification. If not available, the next highest magnification will be
        downsampled. Default value is 20. for 20X objective.
    mask_threshold : float
        Exclude tiles containing less than this minimum percent tissue area when a mask is provided.
        Range for this threshold is (0, 1]. Default value of 0.01 includes tiles with at least 1%
        positive mask pixels.

    Returns
    -------
    study : object
        A histomics_stream study object containing the slides defined in paths, and analysis
        plan defined by tile size, tile overlap, and magnification/reading parameters.
    """

    # wrap string or tuple inputs in list and check arguments
    if isinstance(paths, str):
        paths = [paths]
    elif isinstance(paths, tuple):
        paths = [paths]

    # extract names from lists
    names = []
    for path in paths:
        if isinstance(path, str):
            file = os.path.split(path)[1]
        elif isinstance(path, tuple):
            file = os.path.split(path[0])[1]
        else:
            raise ValueError("Invalid path type.")
        names.append(file)

    # fill basic study parameters
    study = {"version": "version-1"}
    study["tile_height"] = t[0]
    study["tile_width"] = t[1]
    slides = study["slides"] = {}

    # add slides to study
    for i, (name, path) in enumerate(zip(names, paths)):
        if isinstance(path, tuple):
            filename = path[0]
        else:
            filename = path
        slide_name = os.path.split(filename)[1]
        slides[name] = {
            "filename": filename,
            "slide_name": slide_name,
            "slide_group": name,
            "chunk_height": chunk[0],
            "chunk_width": chunk[1],
        }

    # apply settings to each slide
    for name, path in zip(names, paths):
        # generate resolution setting function
        find_resolution_for_slide = hs.configure.FindResolutionForSlide(
            study, target_magnification=objective, magnification_source="exact"
        )

        # generate gridding function
        if isinstance(path, tuple):
            tiles_by_grid_and_mask = hs.configure.TilesByGridAndMask(
                study,
                overlap_height=overlap[0],
                overlap_width=overlap[1],
                mask_filename=path[1],
                mask_threshold=mask_threshold,
            )
        else:
            tiles_by_grid_and_mask = hs.configure.TilesByGridAndMask(
                study,
                overlap_height=overlap[0],
                overlap_width=overlap[1],
            )

        # apply functions
        find_resolution_for_slide(study["slides"][name])
        tiles_by_grid_and_mask(study["slides"][name])

    # apply chunking
    if chunk is not None:
        hs.configure.ChunkLocations()(study)

    return study


def inference(
    iterator,
    model_name,
    source=None,
    target=None,
    url="localhost:8001",
    pre=None,
    nchw=False,
    limit=10,
    rest=1e-3,
    timeout=None,
    verbose=False,
):
    """Inference on the data defined in an iterator.

    Parameters
    ----------
    iterator : object
        An iterator such as LargeImagePrefetch that produces (batch, metadata)
        where batch is a numpy.ndarray and metadata is a dictionary describing the
        batch.
    model_name : str
        The name of the model to use for inference. Models must be loaded prior to
        inference.
    source : array_like
        Stain matrix (3x3) for the input slides. Requires a model with a normalization
        layer. Default value is None.
    target : array_like
        Ideal stain matrix (3x3) for normalization. Required if `source` provided.
    url : str
        The url for the triton server grpc port. Default value is `localhost:8001`.
    pre : function
        A preprocessing function to apply to samples emitted from `iterator` prior
        to inference. Default value is None.
    nchw : bool
        Whether to transpose the data from NHWC format to NCHW format. Default
        value is False.
    limit : int
        The maximum number of allowable pending inferences. Default value 10.
    rest : float
        The number of seconds to wait between polling for completed inferences.
        Default value is 1 millisecond.
    timeout : float
        The number of seconds after which an inference times out. Default value of
        None means no timeout.
    verbose : bool
        Update the console with inference progress and statistics.

    Returns
    -------
    features : list of np.ndarray
        Per-tile inference results
    metadata : dict
        A dictionary inference metadata where values are numpy arrays.
        If using histomics stream this will contain file, magnification, and position
        data for each batch produced by `dataset`.
    performance : dict
        A dictionary of time performance data on reading, inference, and inter-process
        communication.
    failed : list
        A list of failed inference requests.
    """

    # perform inference
    batches = []

    # set flag indicating qin stop signal received
    stop = False

    # create requests object
    req = Requests(url, limit)

    # loop until iterator is exhausted
    while True:
        if not stop:
            # draw samples, preprocess, and submit for inference up to limit
            for i in range(limit - len(req.pending)):
                try:
                    t_start = time()
                    sample, metadata = next(iterator)
                    if not ((source is None) and (target is None)):
                        sample = {
                            "input_0": sample,
                            "input_1": np.stack(
                                sample.shape[0] * [np.cast(source, np.float32)],
                                axis=0,
                            ),
                            "input_2": np.stack(
                                sample.shape[0] * [np.cast(target, np.float32)],
                                axis=0,
                            ),
                        }
                    t_stop = time()
                except StopIteration as e:
                    stop = True
                if not stop:
                    if pre is not None:
                        sample = pre(sample)
                    request = {
                        "model_name": model_name,
                        "inputs": sample,
                        "metadata": metadata,
                        "times": {"read_start": t_start, "read_stop": t_stop},
                    }
                    req.insert(request, timeout)
                else:
                    break

        # check pending requests
        completed = req.check(block=False)

        # put completed post-processed requests into queue
        for inference in completed:
            # remove inputs
            del inference["inputs"]

            # place in queue
            batches.append(inference)

        # if done send stop signal
        if len(req.pending) == 0 and stop:
            break

        # sleep
        sleep(rest)

    # failures
    failed = [b for b in batches if not b["success"]]
    batches = [b for b in batches if b["success"]]

    # successful results
    features = (
        [[b["result"][i] for b in batches] for i in range(len(batches[0]["result"]))]
        if len(batches)
        else []
    )
    metadata = (
        {
            k: np.stack([b[k] for batch in batches for b in batch["metadata"]])
            for k in batches[0]["metadata"][0].keys()
        }
        if len(batches)
        else {}
    )
    times = (
        {k: [b["times"][k] for b in batches] for k in batches[0]["times"].keys()}
        if len(batches)
        else {}
    )

    return features, metadata, times, failed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate embeddings for tiled representations of TIFF based "
            "whole-slide images."
        )
    )
    parser.add_argument(
        "input",
        type=str,
        help=(
            "Path to single image or a tab-delimited file of image paths and "
            "optional masks and stain profiles."
        ),
    )
    parser.add_argument("output", type=str, help="Output directory.")
    parser.add_argument(
        "model",
        type=str,
        help="Model name.",
    )
    parser.add_argument(
        "-f",
        "--float",
        dest="float",
        required=False,
        action="store_true",
        help="Serialize features in float32 precision. Default is float16.",
    )
    parser.add_argument(
        "-m",
        "--mask",
        required=False,
        default=None,
        type=str,
        help="Optional path to a tissue mask image for single image input.",
    )
    parser.add_argument(
        "-n",
        "--normalization",
        required=False,
        default=None,
        type=str,
        help="Optional path to an image stain profile for single image input.",
    )
    parser.add_argument(
        "-r",
        "--target",
        required=False,
        default=None,
        type=str,
        help=("Optional target stain profile for Macenko normalization."),
    )
    parser.add_argument(
        "-s",
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip files with existing embeddings in output.",
    )
    parser.add_argument(
        "-a",
        "--address",
        required=False,
        default="localhost:8001",
        type=str,
        help="Triton server address (default localhost:8001)",
    )
    parser.add_argument(
        "-t",
        "--tile",
        required=False,
        default=224,
        type=int,
        help=("Tile size in pixels (default to internal tile size)."),
    )
    parser.add_argument(
        "-o",
        "--overlap",
        required=False,
        default=0,
        type=int,
        help="Tile overlap in pixels (default 0).",
    )
    parser.add_argument(
        "-M",
        "--magnification",
        required=False,
        default=20.0,
        type=float,
        help="Magnification (default 20X objective).",
    )
    parser.add_argument(
        "-i",
        "--icc",
        action="store_false",
        help="Apply ICC correction (default True).",
    )
    parser.add_argument(
        "-b",
        "--batch",
        required=False,
        default=64,
        type=int,
        help=("Batch size (default 64 tiles)."),
    )
    parser.add_argument(
        "-c",
        "--chunk",
        required=False,
        default=4,
        type=int,
        help=("Reach chunk size (default 4 tiles)."),
    )
    parser.add_argument(
        "-p",
        "--prefetch",
        required=False,
        default=4,
        type=int,
        help=("The number of batches to prefetch from disk (default 4)."),
    )
    parser.add_argument(
        "-w",
        "--workers",
        required=False,
        default=32,
        type=int,
        help=("The number of data loader processes (default 32)."),
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # parse model's configurations
    model = TritonModel(args.model, args.address)
    config = model.get_config()
    dtype = (
        np.float32 if config["input"][0]["dataType"] == "TYPE_FP32" else np.uint8
    )  # input data type

    # capture inputs
    if large_image_source_tiff.canRead(args.input):
        if args.mask is not None:
            if not os.path.isfile(args.mask):
                raise FileNotFoundError(
                    f"Mask file {args.mask} for image {args.input} not found."
                )
        if args.normalization is not None:
            if not os.path.isfile(args.normalization):
                raise FileNotFoundError(
                    f"Stain profile file {args.normalization} for image {args.input} not found."
                )
        files = [[args.input, args.mask, args.normalization]]
    else:
        with open(args.input, "r") as f:
            files = [line.strip().split("\t") for line in f]
        for i, f in enumerate(files):
            if not os.path.isfile(f[0]):
                raise FileNotFoundError(f"Image file {f[0]} not found.")
            f = [*f, *(3 - len(f)) * [None]]
            if f[1] is not None:
                if not os.path.isfile(f[1]):
                    raise FileNotFoundError(
                        f"Mask file {f[1]} for image {f[0]} not found."
                    )
            if f[2] is not None:
                if not os.path.isfile(f[1]):
                    raise FileNotFoundError(
                        f"Stain profile file {f[1]} for image {f[0]} not found."
                    )
            files[i] = f

    # optionally skip files with existing embeddings
    def tfr_name(output, file, model, tile, overlap, magnification):
        return os.path.join(
            output,
            f"{os.path.split(file)[1]}.{model}_{tile}_{overlap}_{magnification}X.tfr",
        )

    if args.skip:
        tfrs = [
            tfr_name(
                args.output,
                f[0],
                args.model,
                args.tile,
                args.overlap,
                str(args.magnification),
            )
            for f in files
        ]
        skip = [(f, t) for (f, t) in zip(files, tfrs) if os.path.isfile(t)]
        for f, t in skip:
            print(f"Skipping image {f[0]}, output {t} exists.")
        files = [f for f in files if f not in [s[0] for s in skip]]

    # ensure presence of target stain profile
    if args.target is None and any([f[2] is not None for f in files]):
        raise ValueError("Path to target stain profile not provided.")
    if args.target is not None:
        target = np.load(args.target)
        if target.shape != (3, 3):
            raise ValueError(
                "Target stain profile expected shape (3, 3), " "found {target.shape}."
            )
    else:
        target = None

    # create studies in background while waiting for inference to finish
    with ProcessPoolExecutor(max_workers=1) as pool:
        # create first study in background
        chunk = args.chunk * args.tile - (args.chunk - 1) * args.overlap
        kwargs = {
            "paths": files[0][0] if files[0][1] is None else (files[0][0], files[0][1]),
            "t": (args.tile, args.tile),
            "chunk": (chunk, chunk),
            "overlap": (args.overlap, args.overlap),
            "objective": args.magnification,
            "mask_threshold": 0.01,
        }
        futures = {0: pool.submit(study, **kwargs)}

        # iterate through files and masks
        for i, (file, mask, stain) in enumerate(tqdm(files)):
            # prefetch study for next slide
            if i < len(files) - 1:
                kwargs.update({"paths": file if mask is None else (file, mask)})
                futures[(i + 1) % 2] = pool.submit(study, **kwargs)

            # wait on study completion for current job
            wait([futures[i % 2]])
            try:
                hs_study = futures[i % 2].result()
            except Exception as exc:
                print(f"Inference error {file}: {exc}")
                continue

            precision = np.dtype("float32") if args.float else np.dtype("float16")

            # tile iterator
            iterator = TiffPrefetch(
                study=hs_study,
                dtype=dtype,
                icc=args.icc,
                batch=args.batch,
                prefetch=args.prefetch,
                workers=args.workers,
            )

            # load source stains
            if stain is not None:
                source = np.load(stain)
                if source.shape != (3, 3):
                    raise ValueError(
                        "Image stain profile expected shape (3, 3), found {target.shape}."
                    )
            else:
                source = None

            # inference
            features, metadata, times, failures = inference(
                iterator,
                args.model,
                source=source,
                target=target,
                url=args.address,
                limit=1,
                rest=0.0,
            )

            # concatenate features
            features = np.concatenate(features[0], axis=0)

            # write to tfrecord
            precision = tf.float32 if args.float else tf.float16
            write_record(
                tfr_name(
                    args.output,
                    file,
                    args.model,
                    args.tile,
                    args.overlap,
                    args.magnification,
                ),
                features,
                metadata,
                labels={},
                structured=False,
                precision=precision,
            )


if __name__ == "__main__":
    main()
