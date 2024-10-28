import hashlib
import json
import numpy as np
import os
import pickle
import pooch
import pytest
import tempfile


"""Data for testing and examples are hosted on Google Drive. Pooch is used to download
this data by parsing registry.csv and prefetching these data once per testing session.
Downloads are deleted when tests are completed."""


HASH_KEYS = {
    "tile_height",
    "tile_width",
    "tile_top",
    "tile_left",
    "target_magnification",
}


def registry():
    """Parse the filename, sha256, and hyperlinks for hosted files"""
    table = os.path.join(os.path.dirname(os.path.realpath(__file__)), "registry.csv")
    hashes = {}
    urls = {}
    with open(table, "r") as f:
        for line in f:
            f, h, u = line.split(",")
            hashes[f] = h
            urls[f] = u
    return hashes, urls


class PrefetchPooch(pooch.Pooch):
    """Modify Pooch to for prefetch of hosted files"""

    def prefetch(self, files=None):
        files = self.registry.keys() if not files else files
        for file in files:
            self._assert_file_in_registry(file)
            self.fetch(
                fname=file,
                processor=(
                    pooch.Unzip(extract_dir="./")
                    if os.path.splitext(file)[1] == ".zip"
                    else None
                ),
            )


@pytest.fixture(scope="session")
def data(files=None):
    """This fixture prefetches hosted data. 
    If running with "./launch_test_container.sh", there will be a `TRITON_TMP_DIR` for temporary data storage.
    If not, put it in user cache
    """
    if os.environ.get("TRITON_TMP_DIR"):
        hashes, urls = registry()
        data = PrefetchPooch(
            path=os.environ.get("TRITON_TMP_DIR"),
            base_url="{version}/",
            registry=hashes,
            urls=urls,
        )
        data.prefetch(files)
        yield data
    else:
        hashes, urls = registry()
        data = PrefetchPooch(
            path=str(pooch.os_cache(os.path.join("pooch", "test_data"))),
            base_url="{version}/",
            registry=hashes,
            urls=urls,
        )
        data.prefetch(files)
        yield data
        


def replace_study_path(study, path):
    """replace wsi path within study object"""
    for image in study["slides"].keys():
        study["slides"][image]["filename"] = path
    return study


def tile_pkl(pkl_path, wsi_path):
    """load a tile pkl containing the study kwargs, metadata, and hashed pixels"""
    with open(pkl_path, "rb") as f:
        contents = pickle.load(f)
    kwargs = contents["kwargs"]
    kwargs["study"] = replace_study_path(kwargs["study"], wsi_path)
    return contents["kwargs"], contents["hashed"]


@pytest.fixture
def tiles_icc(data):
    """returns a dict linking hashed tile metadata and hashed icc tile pixels values"""
    _, hashed = tile_pkl(
        data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.hash_icc.pkl"),
        data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs"),
    )
    return hashed


@pytest.fixture
def tiles(data):
    """returns a dict linking hashed tile metadata and hashed tile pixels values"""
    _, hashed = tile_pkl(
        data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.hash_no_icc.pkl"),
        data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs"),
    )
    return hashed


@pytest.fixture
def it_kwargs_icc(data):
    """returns a study to generate a tile iterator with icc correction"""
    kwargs, _ = tile_pkl(
        data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.hash_icc.pkl"),
        data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs"),
    )
    return kwargs


@pytest.fixture
def it_kwargs(data):
    """returns a study to generate a tile iterator with icc correction"""
    kwargs, _ = tile_pkl(
        data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.hash_no_icc.pkl"),
        data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs"),
    )
    return kwargs


def hash_dict(d):
    """generate a hash from a possibly nested dictionary"""

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


def hash_inference(metadata, features):
    meta_hash = [
        hash_dict({k: metadata[k][i] for k in HASH_KEYS})
        for i in range(len(metadata[list(HASH_KEYS)[0]]))
    ]
    return {m: f for m, f in zip(meta_hash, features)}


@pytest.fixture
def inferred(data):
    """returns a dict linking hashed tile metadata and inference values"""
    with open(
        data.fetch(
            "TCGA-AN-A0G0-01Z-00-DX1.svs.EfficientNetV2S.tensorflow_224_0_20X.pkl"
        ),
        "rb",
    ) as f:
        contents = pickle.load(f)
    return hash_inference(contents["metadata"], contents["features"])


def hash_iterator(iterator):
    """build a dictionary linking tile metadata hashes with tile pixel hashes"""
    batched = [(i, [{k: s[k] for k in HASH_KEYS} for s in m]) for (i, m) in iterator]
    linked = {}
    for batch, metadata in batched:
        for tile, m in zip(batch, metadata):
            linked[hash_dict(m)] = hashlib.sha256(tile.tobytes()).hexdigest()
    return linked


def pkl_iterator(path, kwargs):
    """pickle a hashed version of tile iterator outputs"""
    hashed = hash_iterator(TiffPrefetch(**kwargs))
    with open(path, "wb") as f:
        pickle.dump({"kwargs": kwargs, "hashed": hashed}, f, pickle.HIGHEST_PROTOCOL)
