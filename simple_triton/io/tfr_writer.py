from datetime import datetime
from simple_triton.io import slide_keys, tile_keys
from simple_triton.io.tfr_reader import peek, read_record
from simple_triton.io.tfr_transforms import flatten, structure
import numpy as np
import tensorflow as tf


# acceptable types for user-provided metadata
variable_type_list = [bytes, int, float, str, bool]


def _stack_dicts(dicts, keys, axis=0):
    """This concatenates the tf.Tensor entries in a list of dicts to form
    a single dict. 'keys' is provided as the desired set of keys in the output.
    The conatenation axis is also provided."""
    return {k: tf.concat([d[k] for d in dicts], axis=axis) for k in keys}


def _numpyify_dict(d):
    """This converts the values of a dict from tf.Tensor to np.ndarray. This is
    necessary for serialization. Reading a .tfr generates tf.Tensors, but
    serialization requires these to be np.ndarray type."""
    return {k: d[k].numpy() if hasattr(d[k], "numpy") else d[k] for k in d.keys()}


def _format_variable(variable, variable_type_list=variable_type_list):
    """Formats variables to encode as tf.train.Feature.
    Checks variables for compatibility for storing in .tfr file, and formats
    variables as lists of types defined in variable_type_list. Variables can be
    scalars of types in variable_type_list, a list of these scalars, or
    a 1-D or 0-D numpy.ndarray of the numpy types in variable_type_list. 2-D
    ndarray, nested lists, or dicts are not supported. Strings are byte
    encoded automatically.
    """

    # variable is np.ndarray - check dimensions and return as list
    if isinstance(variable, np.ndarray):
        # variable is 2D or greater
        if variable.ndim > 1:
            raise ValueError("np.ndarray variable must be 0 or 1 dimensional")

        # variable is 1D
        if variable.ndim == 1:
            return list(variable)

        # variable is 0-dim - wrap in list
        if variable.ndim == 0:  # variable is scalar
            return [np.expand_dims(variable, axis=0)[0]]

    # variable is scalar in type_list
    elif isinstance(variable, tuple(variable_type_list)):
        # if variable is string, encode to bytes
        if isinstance(variable, str):
            return [variable.encode("utf-8")]
        else:
            return [variable]

    # variable is list - check type and consistency
    elif isinstance(variable, list):
        t = type(variable[0])
        if t in variable_type_list:
            if all([isinstance(v, t) for v in variable]):
                if t is str:
                    return [v.encode("utf-8") for v in variable]
                else:
                    return variable
            else:
                raise ValueError("Variable (list-type) contains mixed types.")
        else:
            raise ValueError("Variable contains incompatible type.")

    # variable is not np.array, scalar, or list of compatible scalars
    else:
        raise ValueError(
            f"Variable must np.array, or instance / list of compatible types."
        )


def _train_variable(variable, variable_type_list=variable_type_list, serialize=False):
    """Generates a tf.train.Feature depending on the type of input."""

    # format variable
    variable = _format_variable(variable, variable_type_list)

    if serialize:
        variable = [tf.io.serialize_tensor(variable).numpy()]

    # assign to variable conditional on type
    if isinstance(variable[0], bytes):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[v for v in variable])
        )
    elif isinstance(variable[0], (float, np.float32)):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[v for v in variable])
        )
    elif isinstance(variable[0], (int, np.integer, bool)):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[v for v in variable])
        )
    else:
        raise ValueError(
            "Variable must be np.ndarray, list, or instance of bytes, int, float, or bool."
        )


def _generate_tile_index(tile_info):
    """For an inference result containing multiple slides, extract the
    indices of the tiles for each slide as well as the slide name."""

    # format 'slide_name' for input to tf.unique (1D)
    slide_names = tf.squeeze(tf.constant(tile_info["slide_name"]))

    # get unique slide names in result and slide index for each tile
    slide_names, slide_index = tf.unique(slide_names)

    return slide_index, slide_names


def inference_metadata(tile_info):
    """Extracts slide and tile metadata from the inference result of a histomics_stream study.
    histomics_stream produces a dictionary describing the tiles generated
    from the whole slide image. This function parses this data to extract and standardize
    information for storage in a .tfr file. This information includes slide filenames, slide
    groupings, scan magnification, magnification for reading tiles, resized tile magnification,
    tile size, tile overlap, and tile position information.
    Parameters
    ----------
    tile_info : dict
        A dictionary of file, magnification, and position data for each tile produced
        by histomics_stream.
    Returns
    -------
    slide_metadata : dict
        A dict describing the names, groupings, and scan, read, and analysis magnification
        for each slide in the inference result, as well as the tile size and tile overlap
        used for analysis. See .tfr dataset format documentation for details.
    tile_metadata : dict
        A dict describing the position of each tile in the whole-slide image at the analysis
        magnification. See .tfr dataset format documentation for details.
    slide_index : int
        A 1d tensor containing the slide index for each tile. See .tfr dataset format
        documentation for details.
    """

    # get list of unique slide names and tile indices
    slide_index, slide_names = _generate_tile_index(tile_info)

    # sequence of unique slide indices
    seq = tf.range(0, tf.shape(slide_names))

    # for each slide, extract slide metadata
    slide_metadata = {
        k: (np.array([tile_info[k][slide_index == i][0] for i in seq]).ravel())
        for k in slide_keys
        if k in tile_info
    }

    #######################################################
    # Note: this is a patch until histomics stream is fixed
    slide_metadata["level"] = slide_metadata["level"].astype(np.int32)
    #######################################################

    # extract tile metadata - skip 'slide_index' not returned by histomics_stream
    tile_metadata = {k: tile_info[k].ravel() for k in tile_keys if k != "slide_index"}

    return slide_metadata, tile_metadata, slide_index


def _write(
    path, features, slide_index, slide_metadata, tile_metadata, labels, precision
):
    """This function writes the contents of a .tfr. There are multiple file-writing
    functions that produce .tfr files, so we maintain this as a separate function to
    avoid duplicating code."""

    # save feature shape before serialization
    shape = features.shape

    # Adjust floating point precision
    features = tf.cast(features, precision)

    # build file metadata proto
    file_proto = {
        "created": _train_variable(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
        "tf_version": _train_variable(tf.__version__),
    }

    # build slide metadata proto
    slide_proto = {k: _train_variable(slide_metadata[k]) for k in slide_metadata.keys()}

    # build tile metadata proto
    tile_proto = {k: _train_variable(tile_metadata[k]) for k in tile_metadata.keys()}

    # build labels dictionary - prefix added to avoid collision between user-defined labels and other keys
    label_proto = {"label_" + k: _train_variable(labels[k]) for k in labels.keys()}

    # generate protobuffer
    if isinstance(precision, tf.DType):
        size = precision.size
    elif precision in [np.float16, np.float32]:
        size = np.dtype(precision).itemsize
    else:
        raise ValueError("precision must be a tensorflow or numpy dtype.")
    record = {
        "features": _train_variable(tf.reshape(features, [-1]).numpy(), serialize=True),
        "shape": _train_variable(tf.reshape(shape, [-1]).numpy()),
        "slide_index": _train_variable(tf.reshape(slide_index, [-1]).numpy()),
        "precision": _train_variable(tf.reshape(8 * precision.size, [-1]).numpy()),
        **file_proto,
        **slide_proto,
        **tile_proto,
        **label_proto,
    }
    proto = tf.train.Example(features=tf.train.Features(feature=record))

    # write tfrecord file
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(proto.SerializeToString())


def write_labels(path, labels, structured=False, precision=tf.float16):
    """Write labels to an existing .tfr file.

    This overwrites any labels present in the file. Pass an empty dictionary to remove
    labels completely.

    Parameters
    ----------
    path : string
        Path and filename for the generated .tfr file
    labels : dict
        Dictionary containing labels in the form of scalars of types in
        variable_type_list, a list of these scalars, or a 1d or 0d numpy.ndarray
        containing numpy types in variable_type_list. 2d ndarray, nested lists, or
        dicts are not supported. Strings are converted to byte automatically.
    structured : bool
        Flag indicating whether to read features in structured (True)
        or flattened (False) format. Default value is False.
    precision : tensorflow.dtype or numpy.dtype
        Dtype of the stored feature tensor. Features can be stored in either
        16-bit float (default) or 32-bit float. Default value is tf.float16.

    Notes
    -----
    `precision` is required because TensorFlow does not allow this to be inferred at
    runtime.
    """

    # read contents
    serialized = list(tf.data.TFRecordDataset(path))[0]
    variables = peek(serialized)
    features, _, slide_metadata, tile_metadata = read_record(
        serialized, variables, structured=structured, precision=precision
    )

    # convert label values to numpy arrays
    slide_metadata = _numpyify_dict(slide_metadata)
    tile_metadata = _numpyify_dict(tile_metadata)
    labels = _numpyify_dict(labels)

    # remove slide_index from tile_metadata
    slide_index = tile_metadata["slide_index"]
    del tile_metadata["slide_index"]

    # write file to disk
    _write(
        path, features, slide_index, slide_metadata, tile_metadata, labels, precision
    )


def merge_records(path, inputs, label_mode="first", stack_axis=0):
    """This function merges the contents of several .tfr files into a single .tfr file.
    Merged records will automatically be stored in flattened format since structured
    tensors have different dimensions and cannot be trivially concatenated. Several modes
    for handling labels are provided.
    Parameters
    ----------
    path : string
        Path and filename for the generated .tfr file
    inputs : list of string
        A list of the paths/filenames of the .tfr files to merge.
    label_mode : string or dict
        This determines how the labels from multiple files are handled. If
        `first`, only the labels from the first file will be retained. If `stack`,
        the labels from each file will be stacked, keeping only those labels common
        to all files. If a new dict of labels is provided, this will replace the
        existing labels.
    stack_axis : int
        The axis along which to stack labels if `label_mode` is `stack`.
    Notes
    -----
    This function deals with numpy arrays and python commands and is intended to run
    in eager mode for offline data preprocessing. Other reading functions are optimized
    for graph mode and intended for use in training or inference.
    """

    # read in the list of files and capture contents
    features = []
    labels = []
    slide_metadata = []
    tile_metadata = []
    for file in inputs:
        # get serialized contents
        serialized = list(tf.data.TFRecordDataset(file))[0]

        # peek variables
        variables = peek(serialized)

        # read tfr
        f, l, s, t = read_record(serialized, variables, structured=False)

        # capture contents
        features.append(f)
        labels.append(l)
        slide_metadata.append(s)
        tile_metadata.append(t)

    # build combined slide_index - each slide_index ranges from 0 - n_slides-1
    # add cumulative slide count to values of each successive slide_index
    nslides = tf.concat(
        [[tf.reduce_max(t["slide_index"] + 1)] for t in tile_metadata], axis=0
    )
    offset = tf.cumsum(tf.concat([[0], nslides], axis=0))
    slide_index = tf.concat(
        [t["slide_index"] + offset[i] for i, t in enumerate(tile_metadata)], axis=0
    )

    # concatenate features
    features = tf.concat(features, axis=0)

    # stack slide metadata and convert values to numpy arrays
    slide_metadata = _stack_dicts(slide_metadata, slide_metadata[0].keys(), axis=0)
    slide_metadata = _numpyify_dict(slide_metadata)

    # stack tile metadata and convert values to numpy arrays
    tile_metadata = _stack_dicts(tile_metadata, tile_metadata[0].keys(), axis=0)
    tile_metadata["slide_index"] = slide_index
    tile_metadata = _numpyify_dict(tile_metadata)

    # handle labels based on mode - take new label dict, keep labels from first
    # file, or stack labels from all files
    if isinstance(label_mode, dict):
        labels = label_mode

    elif label_mode == "first":
        labels = _numpyify_dict(labels[0])

    elif label_mode == "stack":
        # determine common set of labels in all files
        keys = [list(l.keys()) for l in labels]
        common = keys[0]
        for k in keys:
            common = list(set(common).intersection(set(k)))

        # stack the values for common labels and convert to numpy arrays
        labels = _stack_dicts(labels, common, axis=stack_axis)
        labels = _numpyify_dict(labels)

    else:
        raise ValueError(
            "'label_mode' must be one of 'first', 'stack', or dict containing new label values."
        )

    # write file to disk
    _write(path, features, slide_index, slide_metadata, tile_metadata, labels)


def write_record(
    path, features, tile_info, labels={}, structured=False, precision=tf.float16
):
    """Writes a tfrecord (.tfr) file from the inference results of a histomics stream study.

    Parameters
    ----------
    path : string
        Path and filename for the generated .tfr file.
    features : float
        A two-dimensional feature tensor produced by histomics_stream with instances in rows.
    tile_info : dict
        A dictionary of file, magnification, and position data for each tile produced
        by histomics_stream.
    labels : dict
        Dictionary containing labels in the form of scalars of types in
        variable_type_list, a list of these scalars, or a 1d or 0d numpy.ndarray
        containing numpy types in variable_type_list. 2d ndarray, nested lists, or
        dicts are not supported. Strings are converted to byte automatically. Default
        value is `{}`.
    structured : bool
        Flag indicating whether to read features in structured (True)
        or flattened (False) format. Default value is False.
    precision : tensorflow.dtype or numpy.dtype
        Dtype of the stored feature tensor. Features can be stored in either
        16-bit float (default) or 32-bit float. TensorFlow does not allow this
        to be inferred from the "precision" field of the file at runtime.
        Default value is tf.float16.
    """

    # extract tile coordinates and slide metadata from study
    slide_metadata, tile_metadata, slide_index = inference_metadata(tile_info)

    # reshape features depending on flattened or structured save format
    if structured:
        if len(slide_metadata["slide_name"]) > 1:
            raise ValueError("Cannot write structured output with multiple slides.")
        else:
            tx = slide_metadata["tile_width"]
            ty = slide_metadata["tile_height"]
            x = tile_metadata["tile_left"]
            y = tile_metadata["tile_top"]
            ox = slide_metadata["overlap_width"]
            oy = slide_metadata["overlap_height"]

            # scatter features to structured tensor
            structured_shape = tf.constant(
                [
                    slide_metadata["slide_height_tiles"][0],
                    slide_metadata["slide_width_tiles"][0],
                    features.shape[-1],
                ],
                tf.int32,
            )
            features = structure(features, structured_shape, (x, y), (tx, ty), (ox, oy))

    # write file to disk
    _write(
        path, features, slide_index, slide_metadata, tile_metadata, labels, precision
    )
