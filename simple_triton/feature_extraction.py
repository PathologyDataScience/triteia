import histomics_stream as hs
import numpy as np
import os
from simple_triton.inference import Requests
from time import time


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
        time.sleep(rest)

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
