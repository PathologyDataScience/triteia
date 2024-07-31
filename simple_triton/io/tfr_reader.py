from simple_triton.io import slide_keys, tile_keys
from simple_triton.io.transforms import flatten, structure
import numpy as np
import tensorflow as tf
import warnings


def peek(serialized):
    """Retrieve the variable names and types stored in the serialized contents of
    a .tfr file.

    The return value of peek is required to read .tfr files with TFRecordDataset
    in graph mode. Peek must be called in Eager mode.

    Parameters
    ----------
    serialized : bytes
        The serialized contents of a .tfr file.

    Returns
    -------
    variables : dict
        Key value pairs descibing the names and types (int64_list,
        float_list, byte_list) of all variables stored in the .tfr file.
    """

    # initialize outputs
    variables = {}

    # parse protobuf
    parsed = tf.train.Example()
    if isinstance(serialized, tf.Tensor):
        parsed.ParseFromString(serialized.numpy())
    else:
        parsed.ParseFromString(serialized)

    # iterate over features
    for key, value in parsed.features.feature.items():
        # get feature type
        kind = value.WhichOneof("kind")

        # if duplicate key, raise error
        if key in variables:
            raise ValueError(
                "Duplicate key encountered .tfr - loaded result will be incomplete."
            )

        # capture feature information
        variables[key] = kind

    return variables


def read_record(serialized, variables, structured=False, precision=tf.float16):
    """Reads the contents of a .tfr and returns features, labels, and slide and tile metadata.
    Features can be read in either structured or flattened format, regardless of the format they
    were stored in. Format conversion adds some overhead.

    Parameters
    ----------
    serialized : bytes
        The serialized contents of a .tfr file.
    variables : dict
        Key value pairs descibing the names and types (int64_list,
        float_list, byte_list) of all variables stored in the .tfr file.
    structured : bool
        Flag indicating whether to read features in structured (True)
        or flattened (False) format. Default value is True.
    precision : tensorflow.dtype
        Dtype of the stored feature tensor. Features can be stored in either
        16-bit float (default) or 32-bit float. TensorFllow does not allow this
        to be inferred from the "precision" field of the file at runtime.
        Default value is tf.float16.

    Returns
    -------
    features : float
        A structured ([m, n, D]) or flattened ([m*n, D]) tensor containing
        D-dimensional features for m*n instances.
    labels : dict
        A dictionary containing user-provided metadata that was stored in the
        .tfr file.
    slide_metadata : dict
        A dictionary containing standardized slide metadata produced by histomics_stream.
    tile_metadata : dict
        A dictionary containing standardized tile metadata produced by histomics_stream..
    """

    # Note: contents is a key : value pair produced by io.reader.peek,
    # describing the names and types (bytes_list, float_list, int64_list)
    # of stored variables.

    # mapping of feature types to tf dtypes
    mapping = {
        "bytes_list": tf.string,
        "float_list": tf.float32,
        "int64_list": tf.int64,
    }

    # create description
    description = {k: tf.io.VarLenFeature(mapping[variables[k]]) for k in variables}

    # read in tensors to dict
    contents = tf.io.parse_single_example(serialized, description)
    contents = {
        k: tf.sparse.to_dense(contents[k])
        for k in contents
        if isinstance(contents[k], tf.sparse.SparseTensor)
    }
    contents["features"] = tf.squeeze(contents["features"], 0)

    # decode
    contents["features"] = tf.io.parse_tensor(contents["features"], out_type=precision)

    # extract slide, tile metadata
    for k in slide_keys:
        if k not in contents:
            warnings.warn(f"slide variable {k} missing from .tfr")
    for k in tile_keys:
        if k not in contents:
            warnings.warn(f"tile variable {k} missing from .tfr")
    slide_metadata = {k: contents[k] for k in slide_keys if k in contents}
    tile_metadata = {k: contents[k] for k in tile_keys if k in contents}

    # all other keys are user-provided - trim 'label_' prefix from each key
    label_list = [k for k in contents.keys() if len(k) >= 6]
    label_list = [
        k
        for k in label_list
        if (k[:6] == "label_") and (k not in [*slide_keys, *tile_keys])
    ]
    labels = {k[6:]: contents[k] for k in label_list}

    # extract non-metadata variables
    shape = contents["shape"]
    features = tf.reshape(contents["features"], shape)

    # calculate shape of structured output
    structured_shape = tf.concat(
        [
            slide_metadata["slide_height_tiles"],
            slide_metadata["slide_width_tiles"],
            tf.cast(tf.expand_dims(tf.shape(features)[-1], 0), tf.int64),
        ],
        axis=0,
    )
    structured_shape = tf.cast(structured_shape, tf.int32)

    # get tile size, calculate x, y, overlaps
    tx = contents["tile_width"]
    ty = contents["tile_height"]
    x = tile_metadata["tile_left"]
    y = tile_metadata["tile_top"]
    ox = slide_metadata["overlap_width"]
    oy = slide_metadata["overlap_height"]

    # raise error if .tfr is stored as structured and contains more than one slide
    tf.Assert(
        tf.logical_not(
            tf.logical_and(
                tf.size(shape) == 3, tf.size(slide_metadata["read_magnification"]) > 1
            )
        ),
        [tf.size(shape), tf.size(slide_metadata["read_magnification"])],
    )

    # raise error if structured output is requested and .tfr contains more than one slide
    tf.Assert(
        tf.logical_not(
            tf.logical_and(
                structured, tf.size(slide_metadata["read_magnification"]) > 1
            )
        ),
        [structured, tf.size(slide_metadata["read_magnification"])],
    )

    # structure - structured output requested but features is flattened
    features = tf.cond(
        tf.logical_and(structured, tf.size(shape) == 2),
        lambda: structure(features, structured_shape, (x, y), (tx, ty), (ox, oy)),
        lambda: features,
    )

    # flattened output requested but features is structured
    features = tf.cond(
        tf.logical_and(not structured, tf.size(shape) == 3),
        lambda: flatten(features, (x, y), (tx, ty), (ox, oy)),
        lambda: features,
    )

    return features, labels, slide_metadata, tile_metadata