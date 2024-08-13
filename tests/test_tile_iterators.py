import numpy as np
import os
import pooch
import pytest
from simple_triton.feature_extraction import study
from simple_triton.tile_iterators import TiffPrefetch


# download whole slide image and corresponding mask
wsi_path = pooch.retrieve(
    fname="TCGA-AN-A0G0-01Z-00-DX1.svs",
    url="https://drive.usercontent.google.com/download?id=19agE_0cWY582szhOVxp9h3kozRfB4CvV&export=download&confirm=t",
    known_hash="d046f952759ff6987374786768fc588740eef1e54e4e295a684f3bd356c8528f",
    path=str(pooch.os_cache("pooch")) + os.sep + "wsi",
)
mask_path = pooch.retrieve(
    fname="TCGA-AN-A0G0-01Z-00-DX1.mask.png",
    url="https://drive.usercontent.google.com/download?id=17GOOHbL8Bo3933rdIui82akr7stbRfta&export=download&confirm=t",
    known_hash="bb657ead9fd3b8284db6ecc1ca8a1efa57a0e9fd73d2ea63ce6053fbd3d65171",
    path=str(pooch.os_cache("pooch")) + os.sep + "wsi",
)


def test_nchw():
    """Ensure that transposed nchw/nhwc batches have the same contents"""

    # slide parameters
    tile = 224
    batch = 64
    magnification = 20.0
    chunk = 896
    mask_threshold = 0.5
    prefetch = 4
    workers = 16
    icc = True

    # create a histomics-stream study from a wsi/mask pair
    hs_study = study(
        (wsi_path, mask_path),
        t=(tile, tile),
        chunk=(chunk, chunk),
        objective=magnification,
        mask_threshold=mask_threshold,
    )

    # create tile iterators and pull tiles
    iterator = TiffPrefetch(hs_study, np.uint8, False, icc, batch, prefetch, workers)
    nhwc = [t for (t, m) in iterator]
    iterator = TiffPrefetch(hs_study, np.uint8, True, icc, batch, prefetch, workers)
    nchw = [t for (t, m) in iterator]
    for f, t in zip(nhwc, nchw):
        assert np.array_equal(f.view(), np.transpose(t.view(), [0, 2, 3, 1]))
