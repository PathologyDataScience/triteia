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


def _nested_replace(inbound, replacement):
    if isinstance(inbound, list):
        inbound = [_nested_replace(l, replacement) for l in inbound]
    elif isinstance(inbound, str):
        if inbound in replacement.keys():
            inbound = replacement[inbound]
    return inbound


def _tf_rename_inputs(model, input_index=1):
    """Renames tensorflow model input names for consistency with triton.

    The first input will be renamed as "input_{input_index}". Subsequent inputs will be
    named incrementally.

    Parameters
    ----------
    model : tf.keras.Model
        A model object to rename.
    input_index : int
        The starting index for input layers. Default value is 1.

    Returns
    -------
    renamed : tf.keras.Model
        A model object with inputs renamed in order starting with "input_{input_index}".

    Notes
    -----
    Renaming is applied independently to inputs and outputs for flexibility in combining
    model objects. Renaming cannot be applied to nested models, where a layer is really
    a model containing additional layers.
    """

    # build list of input layer names to replace and replacement names
    config = model.get_config()
    inputs = [
        layer["name"]
        for layer in config["layers"]
        if layer["class_name"] == "InputLayer"
    ]
    replace = {
        i: r
        for i, r in zip(
            inputs,
            [f"input_{i}" for i in range(input_index, len(inputs) + input_index)],
        )
    }

    # modify names instances in layers
    for layer in config["layers"]:
        if layer["class_name"] == "InputLayer":
            layer["name"] = replace[layer["name"]]
            layer["config"]["name"] = replace[layer["config"]["name"]]

    # modify all downstream layers directly connected to these layers
    for layer in config["layers"]:
        if "inbound_nodes" in layer.keys():
            layer["inbound_nodes"] = _nested_replace(layer["inbound_nodes"], replace)

    # replace name instances in config
    config["input_layers"] = _nested_replace(config["input_layers"], replace)

    # transfer to new model
    renamed = tf.keras.Model().from_config(config)
    for n, o in zip(renamed.layers, model.layers):
        n.set_weights(o.get_weights())
    return renamed


def _tf_rename_outputs(model, output_index=1):
    """Renames tensorflow model output names for consistency with triton.

    The first output will be renamed as "output_{output_index}". Subsequent outputs
    will be named incrementally.

    Parameters
    ----------
    model : tf.keras.Model
        A model object to rename.
    output_index : int
        The starting index for output layers. Default value is 1.

    Returns
    -------
    renamed : tf.keras.Model
        A model object with inputs renamed in order starting with "output_{output_index}".

    Notes
    -----
    Renaming is applied independently to inputs and outputs for flexibility in combining
    model objects. Renaming cannot be applied to nested models, where a layer is really
    a model containing additional layers.
    """

    # build list of output layer names
    config = model.get_config()
    outputs = [o[0] for o in config["output_layers"]]
    replace = {
        o: r
        for o, r in zip(
            outputs,
            [f"output_{i}" for i in range(output_index, len(outputs) + output_index)],
        )
    }

    # modify names instances in layers
    for layer in config["layers"]:
        if layer["name"] in replace.keys():
            layer["name"] = replace[layer["name"]]
            layer["config"]["name"] = replace[layer["config"]["name"]]

    # replace name instances in config
    config["output_layers"] = _nested_replace(config["output_layers"], replace)

    # transfer to new model
    renamed = tf.keras.Model().from_config(config)
    for n, o in zip(renamed.layers, model.layers):
        n.set_weights(o.get_weights())
    return renamed


def tf_extractor(
    repository, model, name, input_shape=(224, 224, 3), pooling="avg", normalize=False
):
    """Creates a tensorflow feature extractor model in savedmodel format.

    The model is saved in a folder structure that is compliant with the Triton model
    repository. The base model folder is named with a ".tensorflow" suffix, and an
    additional model version folder is created to store the model files.

    Parameters
    ----------
    repository : str
        Path to the model repository folder where the model will be saved.
    model : str
        Name of model from tf.keras.applications. Currently the modules supported
        include convnext, efficientnet_v2, and resnetv2.
    name : str
        Designated model name for saving in model repository.
    input_shape : tuple(int, int, int)
        The height, width, and channels of tiles used for feature extraction at the
        target magnification. Default value is (224, 224, 3).
    pooling : str {"avg", "max"}
        The pooling mode for the terminal layer of the feature extractor network.
    normalize : bool
        Include a deconvolution-based color normalization layer at the input. Requires
        providing stain matrices for both the input and ideal color profile. Default
        value is False.

    Returns
    -------
    D : int
        Dimensionality of model output.
    """

    # fail if name exists in model repository
    if os.path.exists(os.path.join(repository, name)):
        raise ValueError(f"Model with name {name} already exists in repository.")

    # switch on `model`
    if model.lower() == "efficientnetv2s":
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "efficientnetv2m":
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "efficientnetv2l":
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "convnextbase":
        model = tf.keras.applications.convnext.ConvNeXtBase(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "convnextlarge":
        model = tf.keras.applications.convnext.ConvNeXtLarge(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "convnextsmall":
        model = tf.keras.applications.convnext.ConvNeXtSmall(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "convnexttiny":
        model = tf.keras.applications.convnext.ConvNeXtTiny(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "convnextxlarge":
        model = tf.keras.applications.convnext.ConvNeXtXLarge(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "resnet101v2":
        model = tf.keras.applications.resnet_v2.ResNet101V2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "resnet152v2":
        model = tf.keras.applications.resnet_v2.ResNet152V2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    elif model.lower() == "resnet50v2":
        model = tf.keras.applications.resnet_v2.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=pooling,
        )
    else:
        raise ValueError("model not recognized.")

    # get dimensionality of extracted features
    D = model.output_shape[-1]

    # create the output folder
    os.mkdir(os.path.join(repository, name))
    os.mkdir(os.path.join(path, "1"))
    os.mkdir(os.path.join(path, "model.savedmodel"))

    # rename model inputs and outputs
    model = _tf_rename_outputs(_tf_rename_inputs(model))

    # save model
    model.save(path)

    return D
