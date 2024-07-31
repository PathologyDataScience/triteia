import tensorflow as tf


def _xy_to_ind(x, t, overlap):
    """Converts tile pixel coordinates to row, column indices for
    structured three-dimensional tensor.
    Parameters
    ----------
    x : tuple(int, int)
        The (x, y) coordinates (pixels) of the tile to calculate the index for.
    t : tuple(int, int)
        The tile sizes (pixels).
    overlap : tuple(int, int)
        The tile overlaps (pixels).
    Returns
    -------
    j : int
        The horizontal index of the tile.
    i : int
        The vertical index of the tile.
    """

    j = tf.cast(x[0] / (t[0] - overlap[0]), tf.int32)
    i = tf.cast(x[1] / (t[1] - overlap[1]), tf.int32)

    return j, i


def parallel_dataset(dense_dataset, D, devices, structured=False):
    """Generates a ragged dataset to support multi-GPU training.
    Flattened and structured datasets have variable dimensions that change with
    slide dimensions and/or masking. Multi-GPU training requires creating batches
    that are ragged along these variable dimensions. This function performs
    this ragged transformation and sets shapes to ensure compatibility with keras
    models. A ragged batched dataset is returned.
    Parameters
    ----------
    dense_dataset : tf.data.Dataset
        An unbatched dataset of dense tensors and labels.
    D : int
        The number of features per instance. This is used to define shape[-1]
        of the tensors in dense_dataset. The shape setting operations cannot
        infer this at runtime in graph mode, so it must be provided.
    devices : int
        The number of devices to use in parallel training. Required for batching.
    structured : bool
        Flag indicating whether to read features in structured (True)
        or flattened (False) format. Default value is False.
    Returns
    -------
    parallel_dataset : tf.data.Dataset
    Notes
    -----
    dense_dataset should have len(dense_dataset.element_spec) > 1.

    Since partial batches are discarded, the input dataset should be shuffled
    to avoid discarding the same samples each epoch.
    """

    # set dataset rank
    rank = tf.cond(structured, lambda: 3, lambda: 2)

    # map ensure shape across elements
    dense_dataset = dense_dataset.map(
        lambda x, *args: (tf.ensure_shape(x, [*(rank - 1) * [None], D]), *args)
    )

    # transform to ragged dataset
    parallel_dataset = dense_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(
            batch_size=devices, drop_remainder=True
        )
    )

    return parallel_dataset


def flatten(structured, x, t, overlap):
    """Flattens a structured feature array.
    This transforms an [m, n, D] structured tensor of m*n instances each with
    D features into an [m*n, D] flattened tensor. The order of the rows of the
    flattened tensor depend on their ordering in x, which is not assumed to
    follow any order.
    Parameters
    ----------
    structured : float
        An [m, n, D] structured tensor containing m * n instances each with D features.
    x : tuple(int, int)
        The (x, y) coordinates (pixels) of the upper left corner of each tile
        in the whole slide image.
    t : tuple(int, int)
        The tile sizes (pixels).
    overlap : tuple(int, int)
        The tile overlaps (pixels).
    Returns
    -------
    flattened : float
        An [m * n, D] flattened tensor with rows in order determined by x.
    """

    # calculate tensor x, y indices from tile locations
    j, i = _xy_to_ind(x, t, overlap)

    # gather
    flattened = tf.gather_nd(structured, tf.stack([i, j], axis=1))

    return flattened


def structure(flattened, shape, x, t, overlap):
    """Structure a flattened feature array.
    This transforms an [m*n, D] flattened tensor of m*n instances each with
    D features into an [m, n, D] structured tensor, where instances are organized
    as found in the whole slide image. These positions are calculated using the
    tile sizes, tile overlaps, and pixel coordinates of each tile/instance. Instances
    of the structured tensor that are not defined in x are filled with zeros to
    produce a dense result.
    Parameters
    ----------
    flattened : float
        An [m*n, D] structured tensor containing m * n instances each with D features.
    x : tuple(int, int)
        The (x, y) coordinates (pixels) of the upper left corner of each tile in the
        whole slide image.
    t : tuple(int, int)
        The tile sizes (pixels).
    overlap : tuple(int, int)
        The tile overlaps (pixels).
    Returns
    -------
    structured : float
        An [m, n, D] structured tensor.
    """

    # Notes: features is a two-dimensional array

    # calculate tensor x, y indices from tile locations
    j, i = _xy_to_ind(x, t, overlap)

    # scatter to three-dimensional tensor
    indices = tf.stack([i, j], axis=1)

    # scatter array to
    scattered = tf.scatter_nd(indices, flattened, shape)

    return scattered
