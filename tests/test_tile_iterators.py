from .data import (
    data,
    hash_iterator,
    it_kwargs,
    it_kwargs_icc,
    tiles,
    tiles_icc,
)
import numpy as np
from triteia.tile_iterators import TiffPrefetch


def compare_dict(x, y):
    """ensure equality of two dictionaries containing hashes"""
    assert set(x.keys()) == set(y.keys())
    assert all([x[mhash] == y[mhash] for mhash in x.keys()])


def test_contents_icc_batched(data, it_kwargs_icc, tiles_icc):
    """Compare tile contents to standard with ICC correction"""
    hashed = hash_iterator(TiffPrefetch(**it_kwargs_icc))
    compare_dict(hashed, tiles_icc)


def test_contents_noicc_batched(data, it_kwargs, tiles):
    """Compare tile contents to standard without ICC correction"""
    hashed = hash_iterator(TiffPrefetch(**it_kwargs))
    compare_dict(hashed, tiles)


def test_nonzero(data, it_kwargs):
    """Ensure that tiles are nonzero"""
    batched = [b for (b, m) in TiffPrefetch(**it_kwargs)]
    for b in batched:
        assert not np.array_equal(b.view(), np.zeros(b.shape, b.dtype))


def test_nchw(data, it_kwargs):
    """Ensure that transposed nchw/nhwc batches have the same contents"""
    it_kwargs["nchw"] = False
    nhwc = [t for (t, m) in TiffPrefetch(**it_kwargs)]
    it_kwargs["nchw"] = True
    nchw = [t for (t, m) in TiffPrefetch(**it_kwargs)]
    for f, t in zip(nhwc, nchw):
        assert np.array_equal(f.view(), np.transpose(t.view(), [0, 2, 3, 1]))
