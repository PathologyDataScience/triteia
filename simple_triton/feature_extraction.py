import numpy as np
from simple_triton.inference import Requests


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
    """Inference on the tiles defined in a histomics stream study.

    This shards a study over multiple workers with each worker loading tiles
    and managing the submission and retrieval of different batches of tiles.

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
                    t_put = time.time()
                    sample, metadata = next(dataset)
                    t_get = time.time()
                except StopIteration as e:
                    stop = True
                if not stop:
                    if pre is not None:
                        sample = pre(sample)
                    request = {
                        "model_name": model_name,
                        "inputs": sample,
                        "metadata": metadata,
                        "times": {"qin_put": t_put, "qin_get": t_get},
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
            outputs.append(inference)

        # if done send stop signal
        if len(req.pending) == 0 and stop:
            break

        # sleep
        time.sleep(rest)

    # successful results
    features = [
        [b["result"][i] for b in batches if b["success"]]
        for i in range(len(batches[0]["result"]))
    ]
    metadata = {
        k: np.concatenate([b["metadata"][k] for b in batches if b["success"]])
        for k in batches[0]["metadata"].keys()
    }
    times = {
        k: [b["times"][k] for b in batches if b["success"]]
        for k in batches[0]["times"].keys()
    }

    # failures
    failed = [b for b in batches if not b["success"]]

    return features, metadata, times, failed
