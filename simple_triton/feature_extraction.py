import histomics_stream as hs
import numpy as np
import argparse
import os
import fnmatch
from simple_triton.inference import Requests
from simple_triton.config import ConfigBuilder
from simple_triton.tile_iterators import TiffPrefetch
from simple_triton.model import TritonModel
import large_image_source_tiff
from mil.io.writer import write_record
import tensorflow as tf
from time import sleep, time
from tqdm import tqdm
from pprint import pprint


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
    features = [
        [b["result"][i] for b in batches] for i in range(len(batches[0]["result"]))
    ]
    metadata = {
        k: np.stack([b[k] for batch in batches for b in batch["metadata"]])
        for k in batches[0]["metadata"][0].keys()
    }
    times = {k: [b["times"][k] for b in batches] for k in batches[0]["times"].keys()}

    return features, metadata, times, failed


def main():
    parser = argparse.ArgumentParser(description=("Testing nargs."))
    parser.add_argument(
        "input",
        type=str,
        help="Path to file or file pattern.",
    )
    parser.add_argument("output", type=str, help="Output directory.")
    parser.add_argument(
        "-f",
        "--files",
        required=False,
        type=str,
        help="Path to root folder containing images, or text file listing image paths.",
    )
    parser.add_argument(
        "-n",
        "--noskip",
        dest="skip",
        action="store_false",
        help=("Overwrite existing images (non-default)."),
    )
    parser.add_argument(
        "-s",
        "--server",
        required=False,
        default="localhost:8001",
        type=str,
        help="Triton server address. Default value is `localhost:8001`.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Model name.",
    )
    parser.add_argument(
        "-t",
        "--tile",
        required=False,
        default=None,
        type=int,
        help="Tile size. Defaults to internal file tile size at scan magnification.",
    )
    parser.add_argument(
        "-o",
        "--overlap",
        required=False,
        default=0,
        type=int,
        help="Tile overlap. Defaults to 0 pixels.",
    )
    parser.add_argument(
        "-M",
        "--magnification",
        required=False,
        default=None,
        type=float,
        help="Magnification. Defaults to scan magnification.",
    )
    parser.add_argument(
        "-i",
        "--icc",
        action="store_true",
        help="Apply ICC correction. Defaults to False.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        required=False,
        default=128,
        type=int,
        help=("Batch size. Defaults 128 tiles."),
    )
    parser.add_argument(
        "-p",
        "--prefetch",
        required=False,
        default=4,
        type=int,
        help=("The number of prefetch batches."),
    )
    parser.add_argument(
        "-w",
        "--workers",
        required=False,
        default=32,
        type=int,
        help=("The number of loader processes (default 32)."),
    )
    parser.add_argument(
        "-ma",
        "--mask_dir",
        required=True,
        default=None,
        type=str,
        help=("The directory where the masks are stored"),
    )
    parser.add_argument(
        "-tp",
        "--target_profile",
        required=False,
        default=None,
        type=str,
        help=("Target color profile (.npy file)"),
    )
    parser.add_argument(
        "-sp",
        "--source_profile",
        required=False,
        default=None,
        type=str,
        help=("Path to source color profiles"),
    )

    args = parser.parse_args()

    # check if output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # parse inputs - pattern expansion or file containing list of files
    if isinstance(args.input, list):
        files = [os.path.join(os.getcwd(), f) for f in args.input]
    elif os.path.isdir(args.input):
        files = [os.path.join(args.input, file) for file in os.listdir(args.input)]
    elif os.path.isfile(args.input):
        with open(args.input) as f:
            files = [line.split()[0] for line in f]
    else:
        raise ValueError(
            "Provide one of a text file containing inputs via -f/--files or a file pattern."
        )

    # throw an error when mask directory has not been inputted
    if args.mask_dir is None:
        raise ValueError("Provide the directory where the masks have been stored.")
    else:
        if os.path.isdir(args.mask_dir):
            mask_files = [
                os.path.join(args.mask_dir, mask) for mask in os.listdir(args.mask_dir)
            ]
        else:
            mask_files = [args.mask_dir]

    # target color profile
    if args.target_profile is not None:
        if os.path.isdir(args.target_profile) or not args.target_profile.endswith(
            ".npy"
        ):
            raise ValueError(
                "target profile should be a .npy file containing a stain matrix (3x3)."
            )
        target_p = np.load(args.target_profile)
    else:
        target_p = None

    # source color profiles for slides
    if args.source_profile is not None:
        if os.path.isdir(args.source_profile):
            source_p = [
                os.path.join(args.source_profile, file)
                for file in os.listdir(args.source_profile)
            ]
        elif os.path.isfile(args.source_profile):
            source_p = [args.source_profile]
    else:
        source_p = None

    # match files to masks and source profiles and throw an error if a slide is not matched to a maks/profile
    print("---------------- matching files to masks and source profiles -------------")
    matched_files = []
    for file in files:
        matched_mask = False
        matched_source = False

        for mask in mask_files:
            if fnmatch.fnmatch(mask.split("/")[-1], f'*{file.split("/")[-1]}*'):
                matched_mask = True
                break
        if not matched_mask:
            raise FileNotFoundError(f"No mask file found for {file}")

        if source_p is not None:
            for prof in source_p:
                if fnmatch.fnmatch(prof.split("/")[-1], f'*{file.split("/")[-1]}*'):
                    matched_source = True
                    break
            if not matched_source:
                raise FileNotFoundError(f"No source profile found for {file}")
            matched_files.append((file, mask, np.load(prof)))
        else:
            matched_files.append((file, mask, None))

    # check of model is loaded
    print("---------------- loading model -------------")
    model = TritonModel(args.model, args.server)
    model.unload()
    model.load()
    if model.is_loaded:
        print("The model has already been loaded")
        pprint(model.get_config())
    else:
        try:
            # load tensorflow model - set maximum batch size
            model = TritonModel(args.model, args.server)

            # initialize builder with a basic configuration
            builder = ConfigBuilder(args.model, config={"maxBatchSize": args.batch})

            # increase the number of model instances per GPU to 2
            builder.add_instance_group(count=2)

            # add automatic mixed precision
            builder.add_mixed_precision()

            # re-load model with new config
            model.load(config=builder.config)

            assert model.is_loaded()
        except Exception as e:
            print("loading model failed: " + str(e), flush=True)

    # iterate through files and masks
    for file, mask, source_p in matched_files:
        # start timer
        start = time()

        # determine magnification, tile size if not provided
        source = large_image_source_tiff.open(file)
        metadata = source.getMetadata()
        magnification = (
            metadata["magnification"]
            if args.magnification is None
            else args.magnification
        )
        t = (
            (metadata["tileHeight"], metadata["tileWidth"])
            if args.tile is None
            else (args.tile, args.tile)
        )
        if (args.tile is None) and (magnification != metadata["magnification"]):
            raise ValueError(
                (
                    "Using default tile size requires native magnification "
                    f"`None` or {metadata['magnification']}."
                )
            )

        # create tile source
        hs_study = study(
            (file, mask),
            t=t,
            chunk=t,
            objective=magnification,
        )

        # tile iterator
        iterator = TiffPrefetch(
            hs_study, np.uint8, args.icc, args.batch, args.prefetch, args.workers
        )

        # inference
        features, metadata, times, failures = inference(
            iterator,
            args.model,
            w_source=source_p,
            w_target=target_p,
            url=args.server,
            limit=1,
            rest=0.0,
        )

        # concatenate features
        features = np.concatenate(features[0], axis=0)

        # write to tfrecord
        tfr_file = "{}/{}.{}_{}_{}X.tfr".format(
            args.output,
            file.split("/")[-1],
            args.model,
            t[0],
            args.overlap,
            str(int(magnification)),
        )
        write_record(
            tfr_file,
            features,
            metadata,
            labels={},
            structured=False,
            precision=tf.float16,
        )

        # display elapsed time
        print(
            (
                f"{os.path.split(tfr_file)[1]} - "
                f"{features.shape[0]} tiles, elapsed time: {time()-start}"
            )
        )


if __name__ == "__main__":
    main()
