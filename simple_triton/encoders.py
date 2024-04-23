import os
import tensorflow as tf


def reshape_savedmodel(
    path,
    savedmodel,
    shape,
    dtype=tf.float32,
    signature="serving_default",
    outputs=["output_1"],
):
    """Modify a savedmodel to change input signature.

    Some savedmodel files lack a batch dimension or have input
    dimensions set when they should be variable. This makes hosting
    on Triton difficult. This function generates a copy of the
    savedmodel with an altered input signature.

    Parameters
    ----------
    path : str
        The path for the output savedmodel.
    savedmodel : str
        The path to the savedmodel root folder.
    shape: array-like
        The new input shape signature. Use `None` for variable
        dimensions.
    dtype : tensorflow.python.framework.dtypes.DType
        The output dtype.
    signature : str
        The savedmodel signature to modify and save.
    outputs : array-like
        A list of outputs to capture in the modified savedmodel.

    Notes
    -----
    Models altered using this method may need to be loaded on
    triton with the platform explicitly mentioned in the
    configuration: {"platform": "tensorflow_savedmodel"}.
    """

    # define a class that discards batch dimension
    class Reshaped(tf.Module):
        def __init__(self, model):
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec(shape, dtype)])
        def fn(self, input_1):
            inference = self.model.signatures[signature](input_1[0])
            return {o: inference[o] for o in outputs}

    # instantiate and call model
    model = tf.saved_model.load(savedmodel)
    reshaped = Reshaped(model)
    tf.saved_model.save(
        reshaped,
        path,
        signatures={signature: reshaped.fn},
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=False),
    )


def _nested_replace(inbound, replacement):
    if isinstance(inbound, list):
        inbound = [_nested_replace(l, replacement) for l in inbound]
    elif isinstance(inbound, str):
        if inbound in replacement.keys():
            inbound = replacement[inbound]
    return inbound


def tf_rename_inputs(model, input_index=0):
    """Renames tensorflow model input names for consistency with triton.

    The first input will be renamed as "input_{input_index}". Subsequent inputs will be
    named incrementally.

    Parameters
    ----------
    model : tf.keras.Model
        A model object to rename.
    input_index : int
        The starting index for input layers. Default value is 0.

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


def tf_rename_outputs(model, output_index=0):
    """Renames tensorflow model output names for consistency with triton.

    The first output will be renamed as "output_{output_index}". Subsequent outputs
    will be named incrementally.

    Parameters
    ----------
    model : tf.keras.Model
        A model object to rename.
    output_index : int
        The starting index for output layers. Default value is 0.

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


class TfCast(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float32, **kwargs):
        super(TfCast, self).__init__(**kwargs)
        self.cast = dtype

    def call(self, inputs):
        return tf.cast(inputs, self.cast)

    def get_config(self):
        return super(TfCast, self).get_config()


def tf_encoder(
    repository,
    model,
    name,
    input_shape=(224, 224, 3),
    dtype=tf.uint8,
    pooling="avg",
    normalize=False,
    version=1,
):
    """Creates a tensorflow encoder model in savedmodel format.

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

    # remove encoder input/output layers and extract configs
    # add input layer and optional devonvolution layer
    input_layer = model.layers.pop(0)
    input_kwargs = input_layer.get_config()
    output_layer = model.layers.pop(len(model.layers) - 1)
    output_kwargs = output_layer.get_config()
    output_kwargs["name"] = "output_0"
    if "config" in output_kwargs:
        if "name" in output_kwargs["config"]:
            output_kwargs["config"]["name"] = "output_0"
    input_0 = tf.keras.layers.Input(
        shape=input_shape,
        dtype=dtype,
        name="input_0",
        sparse=input_kwargs["sparse"],
        ragged=input_kwargs["ragged"],
    )
    output_0 = type(output_layer).from_config(output_kwargs)
    float_input = TfCast(tf.float32)(input_0) if dtype != tf.float32 else input_0
    if normalize:
        input_1 = tf.keras.layers.Input(shape=[3, 3], name="input_1")
        input_2 = tf.keras.layers.Input(shape=[3, 3], name="input_2")
        deconv = DeconvNorm()([float_input, input_1, input_2])
        model = tf.keras.Model([input_0, input_1, input_2], model(deconv))
    else:
        model = tf.keras.Model(input_0, output_0(model(float_input)))

    # get dimensionality of extracted features
    D = model.output_shape[-1]

    # create the output folder
    path = os.path.join(repository, name, "1", "model.savedmodel")
    os.makedirs(path)

    # save model
    model.save(path)

    return D


def normalize(w):
    """Normalize the columns of a stain matrix to unit-norm.

    Parameters
    ----------
    w : array_like
        A 3x3 array with stain vectors in columns.

    Returns
    -------
    normed : array_like
        A 3x3 array where the columns are unit-norm.
    """

    return tf.divide(w, tf.norm(w, axis=0))


def rgb_to_sda(rgb, i0=256.0, allow_negatives=False):
    """Transform batched rgb image data to stain darkness colorspace.

    Parameters
    ----------
    rgb : array_like
        An NWHC array of batched rgb colorspace image data.
    i0 : float
        Background intensity for all channels. Default value is 256.
    allow_negatives : bool
        Whether to allow negative intensities in SDA space. Default
        value is False.

    Returns
    -------
    sda : array_like
        An NWHC array of batched sda colorspace image data.
    """

    shifted = tf.math.maximum(rgb + 1.0, 1e-10)
    sda = -tf.math.log(shifted / i0) * 255.0 / tf.math.log(i0)
    if not allow_negatives:
        sda = tf.math.maximum(sda, 0.0)
    return sda


def sda_to_rgb(sda, i0=256.0):
    """Transform batched sda image data to rgb colorspace.

    Parameters
    ----------
    sda : array_like
        An NWHC array of batched sda colorspace image data.
    i0 : float
        Background intensity for all channels. Default value is 256.

    Returns
    -------
    rgb : array_like
        An NWHC array of batched rgb colorspace image data.
    """

    return i0 ** (1.0 - sda / 255.0) - 1.0


def matmul_channel(w, batched):
    """Multiply a stain matrix or matrix inverse against batched image data.

    This applies a 3x3 matrix to the channel dimension of an NHWC array
    representing batched image data.

    Parameters
    ----------
    w : array_like
        A 3x3 array representing stains (in columns) or the inverse
        of such a stain matrix.
    batched : array_like
        An NWHC array of batched image data.

    Returns
    -------
    product : array_like
        An NWHC array representing the doc product w * batched along the
        channel dimension.
    """

    return tf.transpose(tf.tensordot(w, batched, axes=[[1], [3]]), perm=[1, 2, 3, 0])


@tf.keras.saving.register_keras_serializable()
class DeconvNorm(tf.keras.layers.Layer):
    """A deconvolution-based color normalization layer.

    This non-trainable layer takes as input an image, a source stain matrix,
    and a target stain matrix, and uses color deconvolution to map the input
    image color profile to the target profile. The image is first deconvolved
    using it's source stain matrix to create a stain concentration image, and
    is then transformed back to rgb space using the target stain matrix. Stain
    matrices can be estimated using the Macenko method in histomicstk. This
    operation works on batched images.

    Parameters
    ----------
    i0 : float
        Background intensity for glass regions. One value for all channels.
        Default value is 256.
    allow_negatives : bool
        Whether to allow negative values in the stain darkness color space.
        Default value is False.

    Attributes
    ----------
    i0 : float
        Background intensity for glass regions. One value for all channels.
        Default value is 256.
    allow_negatives : bool
        Whether to allow negative values in the stain darkness color space.
        Default value is False.

    Methods
    -------
    call(inputs)
        A list containing the input image (float), a 3x3 source stain matrix
        where each column represents a stain (float), and a 3x3 target stain
        matrix representing the ideal color profile (float).
    """

    def __init__(self, i0=256.0, allow_negatives=False, **kwargs):
        super(DeconvNorm, self).__init__(**kwargs)
        self.i0 = i0
        self.allow_negatives = allow_negatives

    def call(self, inputs):
        # unpack and normalize stain matrix inputs
        w_source = normalize(inputs[1][0])
        w_target = normalize(inputs[2][0])

        # invert w_source
        winv_source = tf.linalg.pinv(w_source)

        # conversion from rgb to stain darkness (SDA) space
        sda = rgb_to_sda(inputs[0], self.i0, self.allow_negatives)

        # apply source deconvolution to separate into stains
        deconvolved = matmul_channel(winv_source, sda)

        # clip to limits in sda space that correspond to [0, 255] rgb space
        lower = -tf.math.log(0.0 + 1.0 / self.i0) * 255.0 / tf.math.log(self.i0)
        upper = -tf.math.log(255.0 + 1.0 / self.i0) * 255.0 / tf.math.log(self.i0)
        deconvolved = tf.clip_by_value(
            deconvolved, tf.minimum(lower, upper), tf.maximum(lower, upper)
        )

        # apply target convolution to remix
        convolved = matmul_channel(w_target, deconvolved)

        # conversion from stain darkness to rgb
        rgb = sda_to_rgb(convolved, self.i0)

        return rgb

    def get_config(self):
        config = super().get_config()
        config.update({"i0": self.i0, "allow_negatives": self.allow_negatives})
        return config
