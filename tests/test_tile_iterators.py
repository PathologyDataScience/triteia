from .data import data
import hashlib
import json
import numpy as np
import os
import pickle
import pytest
from simple_triton.feature_extraction import study
from simple_triton.tile_iterators import TiffPrefetch


HASH_KEYS = {
    "tile_height",
    "tile_width",
    "tile_top",
    "tile_left",
    "target_magnification",
}


def hash_dict(d):
    """hash a possibly nested dictionary"""

    def immutify(d):
        d_new = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d_new[k] = tuple(v.tolist())
            elif isinstance(v, list):
                d_new[k] = tuple(v)
            elif isinstance(v, dict):
                d_new[k] = immutify(v)
            else:
                # convert numpy types to native
                if hasattr(v, "dtype"):
                    d_new[k] = v.item()
                else:
                    d_new[k] = v
        return dict(sorted(d_new.items(), key=lambda item: item[0]))

    d_hashable = immutify(d)
    s_hashable = json.dumps(d_hashable).encode("utf-8")
    m = hashlib.sha256(s_hashable).hexdigest()
    return m


def compare_dict(x, y):
    """ensure equality of two dictionaries containing hashes"""
    assert set(x.keys()) == set(y.keys())
    assert all([x[mhash] == y[mhash] for mhash in x.keys()])


def meta_tile_dict(batched):
    """build a dictionary linking tile metadata hashes with tile pixel hashes"""
    linked = {}
    for batch, metadata in batched:
        for tile, m in zip(batch, metadata):
            linked[hash_dict(m)] = hashlib.sha256(tile.tobytes()).hexdigest()
    return linked


def pkl_tiles(path, kwargs):
    """pickle a hashed version of tile iterator outputs"""
    iterator = TiffPrefetch(**kwargs)
    keys = {
        "tile_height",
        "tile_width",
        "tile_top",
        "tile_left",
        "target_magnification",
    }
    batches = [(i, [{k: s[k] for k in keys} for s in m]) for (i, m) in iterator]
    hashed = meta_tile_dict(batches)
    with open(path, "wb") as f:
        pickle.dump({"kwargs": kwargs, "hashed": hashed}, f, pickle.HIGHEST_PROTOCOL)


def unpkl_tiles(path):
    """load hashed version of tile iterator outputs"""
    with open(path, "rb") as f:
        contents = pickle.load(f)
    for image in contents["kwargs"]["study"]["slides"].keys():
        filename = os.path.split(
            contents["kwargs"]["study"]["slides"][image]["filename"]
        )[1]
        filename = os.path.join(os.path.split(path)[0], os.path.split(filename)[1])
        if os.path.isfile(filename):
            contents["kwargs"]["study"]["slides"][image]["filename"] = filename
        else:
            raise Exception(f"Test image {filename} not found.")
    return contents["kwargs"], contents["hashed"]


def test_contents_icc_batched(data):
    """Compare tile contents to gold-standard with ICC correction"""
    kwargs, hashed = unpkl_tiles(data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.hash_icc.pkl"))
    iterator = TiffPrefetch(**kwargs)
    batches = [(i, [{k: s[k] for k in HASH_KEYS} for s in m]) for (i, m) in iterator]
    compare_dict(meta_tile_dict(batches), hashed)


def test_contents_noicc_batched(data):
    """Compare tile contents to gold-standard without ICC correction"""
    kwargs, hashed = unpkl_tiles(
        data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.hash_no_icc.pkl")
    )
    iterator = TiffPrefetch(**kwargs)
    batches = [(i, [{k: s[k] for k in HASH_KEYS} for s in m]) for (i, m) in iterator]
    compare_dict(meta_tile_dict(batches), hashed)


def test_nonzero(data):
    """Ensure that tiles are nonzero"""
    kwargs, _ = unpkl_tiles(data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.hash_icc.pkl"))
    iterator = TiffPrefetch(**kwargs)
    batched = [b for (b, m) in iterator]
    for b in batched:
        assert not np.array_equal(b.view(), np.zeros(b.shape, b.dtype))


def test_nchw(data):
    """Ensure that transposed nchw/nhwc batches have the same contents"""
    kwargs, _ = unpkl_tiles(data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.hash_icc.pkl"))
    kwargs["nchw"] = False
    iterator = TiffPrefetch(**kwargs)
    nhwc = [t for (t, m) in iterator]
    kwargs["nchw"] = True
    iterator = TiffPrefetch(**kwargs)
    nchw = [t for (t, m) in iterator]
    for f, t in zip(nhwc, nchw):
        assert np.array_equal(f.view(), np.transpose(t.view(), [0, 2, 3, 1]))
