import os
import tensorflow as tf


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
    
    # fail if name exists in model repository
    if os.path.exists(os.path.join(repository, name)):
        raise ValueError(f"Model with name {name} already exists in repository.")
        
    # switch on `model`
    if model.lower() == "efficientnetv2s":
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)        
    elif model.lower() == "efficientnetv2m":
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)
    elif model.lower() == "efficientnetv2l":
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)
    elif model.lower() == "convnextbase":
        model = tf.keras.applications.convnext.ConvNeXtBase(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)
    elif model.lower() == "convnextlarge":
        model = tf.keras.applications.convnext.ConvNeXtLarge(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)
    elif model.lower() == "convnextsmall":
        model = tf.keras.applications.convnext.ConvNeXtSmall(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)
    elif model.lower() == "convnexttiny":
        model = tf.keras.applications.convnext.ConvNeXtTiny(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)
    elif model.lower() == "convnextxlarge":
        model = tf.keras.applications.convnext.ConvNeXtXLarge(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)
    elif model.lower() == "resnet101v2":
        model = tf.keras.applications.resnet_v2.ResNet101V2(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)        
    elif model.lower() == "resnet152v2":
        model = tf.keras.applications.resnet_v2.ResNet152V2(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)
    elif model.lower() == "resnet50v2":
        model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False, weights='imagenet', input_shape=(t[0], t[1], 3),
        pooling=pool)    
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

    # save model
    model.save(path)

    return D
