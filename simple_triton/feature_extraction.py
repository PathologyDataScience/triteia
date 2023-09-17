import multiprocessing
import multiprocessing.queues
import numpy as np
import os
from simple_triton.inference import InferenceRunner
from simple_triton.sharded_tiles import ShardedTiles
from simple_triton.utils import TimedQueue
import tensorflow as tf


def histomics_stream_inference(
    study,
    model_name,
    url="localhost:8001",
    batch=64,
    workers=32,
    limit=10,
    pre=None,
    transpose=False,
):
    """Inference on the tiles defined in a histomics stream study.

    This shards a study over multiple workers with each worker loading tiles
    and managing the submission and retrieval of different batches of tiles.

    Parameters
    ----------
    study : dict
        A histomics_stream study object containing the slides defined in paths, and
        analysis plan defined by tile size, tile overlap, and magnification/reading
        parameters. Can contain multiple slides. This study is sharded over multiple
        workers.
    model_name : str
        The name of the model to use for inference. This model should be
        loaded on triton prior to inference.
    url : str
        The url for the triton server grpc port. Default value is `localhost:8001`.
    batch : int
        The number of tiles to process in a batch. Default value is `64` tiles.
        If `0`, inference will be performed on single tiles with no batch
        dimension.
    workers : int
        The number of workers to use for reading tiles from disk and submitting and
        receiving inference results. Each worker will receive a shard of tiles and
        will read them using a ShardedTiles iterator. Default value `32`.
    limit : int
        The maximum number of batches pending inference allowed for each worker.
    pre : function
        A preprocessing function to apply to samples emitted from `dataset` prior
        to inference. Default value is None.
    transpose : bool
        Whether to transpose the data from NHWC format to NCHW format. Default
        value is False.

    Returns
    -------
    features : list of np.ndarray
        Per-tile inference results
    tile_info : dict
        A dictionary of file, magnification, and position data for each tile produced
        by histomics_stream.
    performance : dict
        A dictionary of time performance data on reading, inference, and inter-process
        communication.
    failed : list
        A list of failed inference requests.

    See Also
    --------
    ShardedTiles
    """

    # create input, output queues
    qout = TimedQueue()

    # Start consumers
    shards = []
    for w in range(workers):
        shard = ShardedTiles(study, batch, w, workers, transpose)
        shards.append(
            InferenceRunner(url, model_name, shard, qout, limit, pre=pre, verbose=False)
        )
    for s in shards:
        s.start()

    # collecct results
    batches = []
    N = workers
    while N:
        output, t_put, t_get = qout.get()
        if output is None:
            N -= 1
        else:
            output["times"]["qout_put"] = t_put
            output["times"]["qout_get"] = t_get
            batches.append(output)

    # successful results results
    features = [
        [b["result"][i] for b in batches if b["success"]]
        for i in range(len(batches[0]["result"]))
    ]
    tile_info = {
        k: np.concatenate([b["metadata"][k] for b in batches if b["success"]])
        for k in batches[0]["metadata"].keys()
    }
    times = {
        k: [b["times"][k] for b in batches if b["success"]]
        for k in batches[0]["times"].keys()
    }

    # failures
    failed = [b for b in batches if not b["success"]]

    return features, tile_info, times, failed


def feature_extractor(repository, model, name, t=(224, 224), pool="avg"):
    """Creates a savedmodel format feature extractor model.

    Parameters
    ----------
    repository : str
        Path to the model repository folder on the triton host.
    model : str {}
        Name of model
    name : str
        Designated model name to use for saving in repository.
    t : tuple(int, int)
        The height and width (pixels) of the tiles used for analysis at target magnification.
        Default value is (224, 224).

    Returns
    -------
    D : int
        Dimensionality of model output.
    """

    def rename(model, input_name="input_1", output_name="output_1"):
        # renames feature extraction model input, ouput names to input_1, output_1

        def nested_replace(inbound, name, replacement):
            if isinstance(inbound, list):
                inbound = [nested_replace(l, name, replacement) for l in inbound]
            else:
                if inbound == name:
                    inbound = replacement
            return inbound

        config = model.get_config()
        replace = config["layers"][0]["name"]
        config["layers"][0]["name"] = input_name
        config["layers"][0]["config"]["name"] = input_name
        config["layers"][1]["inbound_nodes"] = nested_replace(
            config["layers"][1]["inbound_nodes"], replace, input_name
        )
        config["input_layers"] = nested_replace(
            config["input_layers"], replace, input_name
        )
        replace = config["layers"][-1]["name"]
        config["layers"][-1]["name"] = output_name
        config["layers"][-1]["config"]["name"] = output_name
        config["output_layers"] = nested_replace(
            config["output_layers"], replace, output_name
        )
        new = tf.keras.Model().from_config(config)
        for n, o in zip(new.layers, model.layers):
            n.set_weights(o.get_weights())
        return new

    # fail if name exists in model repository
    if os.path.exists(os.path.join(repository, name)):
        raise ValueError(f"Model with name {name} already exists in repository.")

    # switch on `model`
    if model.lower() == "efficientnetv2s":
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "efficientnetv2m":
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "efficientnetv2l":
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "convnextbase":
        model = tf.keras.applications.convnext.ConvNeXtBase(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "convnextlarge":
        model = tf.keras.applications.convnext.ConvNeXtLarge(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "convnextsmall":
        model = tf.keras.applications.convnext.ConvNeXtSmall(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "convnexttiny":
        model = tf.keras.applications.convnext.ConvNeXtTiny(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "convnextxlarge":
        model = tf.keras.applications.convnext.ConvNeXtXLarge(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "resnet101v2":
        model = tf.keras.applications.resnet_v2.ResNet101V2(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "resnet152v2":
        model = tf.keras.applications.resnet_v2.ResNet152V2(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    elif model.lower() == "resnet50v2":
        model = tf.keras.applications.resnet_v2.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=(t[0], t[1], 3),
            pooling=pool,
        )
    else:
        raise ValueError("model not recognized.")

    # get dimensionality of extracted features
    D = model.output_shape[-1]

    # create the output folder
    path = os.path.join(repository, name)
    os.mkdir(path)
    path = os.path.join(path, "1")
    os.mkdir(path)
    path = os.path.join(path, "model.savedmodel")
    os.mkdir(path)

    # rename model inputs and outputs
    model = rename(model)

    # save model
    model.save(path)

    return D
