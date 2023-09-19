from copy import deepcopy
import large_image_source_tiff as large_image
import numpy as np


def _byteify(string):
    return bytes(string.encode("utf-8"))


def _txr_keys(dictionary, keys):
    return {
        k: _byteify(dictionary[k]) if isinstance(dictionary[k], str) else dictionary[k]
        for k in keys
        if k in dictionary.keys()
    }


def _hs_study_meta(study):
    keys = ["version", "tile_height", "tile_width", "overlap_height", "overlap_width"]
    return _txr_keys(study, keys)


def _hs_slide_meta(slide, study):
    slide_keys = [
        "filename",
        "slide_name",
        "slide_group",
        "chunk_width",
        "chunk_height",
        "target_magnification",
        "scan_magnification",
        "read_magnification",
        "returned_magnification",
        "level",
        "slide_width",
        "slide_height",
        "slide_height_tiles",
        "slide_width_tiles",
        "mask_height",
        "mask_width",
    ]
    study_keys = ["chunk_width", "chunk_height"]
    return {**_txr_keys(slide, slide_keys), **_txr_keys(study, study_keys)}


def _hs_tile_meta(tile, study):
    return {
        "chunk_top": tile["tile_top"],
        "chunk_left": tile["tile_left"],
        "chunk_bottom": tile["tile_top"] + study["tile_height"],
        "chunk_right": tile["tile_left"] + study["tile_width"],
        "tile_top": tile["tile_top"],
        "tile_left": tile["tile_left"],
    }


def _hs_index(study, indices):
    # discard non-shard tiles from study - indices assumed contiguous
    # but can span multiple slides in a multi-slide study
    filtered = deepcopy(study)
    cumulative = 0
    for slide in study["slides"].keys():
        keys = list(study["slides"][slide]["tiles"].keys())
        keycount = len(keys)
        keys = [keys[i] for i in indices - cumulative if i >= 0 and i < keycount]
        cumulative = cumulative + keycount
        tiles = {tile: study["slides"][slide]["tiles"][tile] for tile in keys}
        filtered["slides"][slide]["tiles"] = tiles
    # delete empty slides
    filtered["slides"] = {
        slide: filtered["slides"][slide]
        for slide in filtered["slides"].keys()
        if len(filtered["slides"][slide]["tiles"]) > 0
    }
    return filtered


class ShardedTiles(object):
    """Iterator for sharded reading with large image.

    This class is used for sharded multiprocessing reading using
    the large_image library. This can be used to improve throughput
    for inference tasks. The large_image object used for reading is
    created on the first read, making ShardedLargeImage object
    serializable.

    Parameters
    ----------
    study : dict
        A study dictionary from histomics_stream, defining the reading
        parameters and tile locations for possibly multiple slides.
    w_source : array_like
        Stain matrix (3x3) for the input slides. Requires a model with a normalization 
        layer. Default value is None.
    w_target : array_like
        Ideal stain matrix (3x3) for normalization. Requires a model with a 
        normalization layer. Default value is None.
    batch : int
        The number of tiles in each batch. Partial batches are not padded. If 0,
        single sample batches will be generated without the singleton batch
        dimension.
    worker_index : int
        The worker index, ranging from 0 to `num_workers`.
    num_workers : int
        The total number of workers.
    nchw : bool
        Transpose the dimensions of tiles to NCHW/CHW format for ONNX and Torch 
        models. Default value is False.

    Returns
    -------
    tiles : array-like
        A four-dimensional BHWC numpy array of batched tiles.
    metadata : dict
        A dictionary of tile position and magnification information.

    Attributes
    ----------
    batch : int
        The number of tiles in each batch. If 0, single sample batches will be
        generated without the singleton batch dimension.
    worker_index : int
        The worker index, ranging from 0 to `num_workers`.
    num_workers : int
        The total number of workers.

    Notes
    -----
    For more details on `metadata` see
    https://github.com/DigitalSlideArchive/HistomicsStream/blob/master/StudyObject.md
    """

    def __init__(
        self, study, w_source, w_target, batch, worker_index, num_workers, nchw=False
    ):
        self.i = 0
        self.study = study
        if batch == 0:
            self._singleton = True
        else:
            self._singleton = False
        self.batch = batch
        self.worker_index = worker_index
        self.num_workers = num_workers
        self._nchw = nchw
        self.large_images = None
        self._shard()

        # ensure that if provided, w_source, w_target are batch x 3 x 3
        if w_source is not None:
            w_source = np.repeat(w_source, [batch, 1, 1], axis=0)
        if w_target is not None:
            w_target = np.repeat(w_target, [batch, 1, 1], axis=0)
        self.w_source = w_source
        self.w_target = w_target

    def _shard(self):
        # generates the large_image read parameters and metadata for shard
        tile_count = sum(
            [
                len(self.study["slides"][slide]["tiles"])
                for slide in self.study["slides"].keys()
            ]
        )
        tile_indices = np.arange(
            tile_count * self.worker_index // self.num_workers,
            tile_count * (self.worker_index + 1) // self.num_workers,
        )
        self.study = _hs_index(self.study, tile_indices)
        self.tiles = [
            (
                slide["filename"],
                {
                    "scale": {"magnification": slide["target_magnification"]},
                    "format": "numpy",
                    "region": {
                        "left": tile["tile_left"],
                        "top": tile["tile_top"],
                        "width": self.study["tile_width"],
                        "height": self.study["tile_height"],
                        "units": "mag_pixels",
                    },
                },
            )
            for slide in self.study["slides"].values()
            for tile in slide["tiles"].values()
        ]
        self.metadata = [
            {
                **_hs_study_meta(self.study),
                **_hs_slide_meta(slide, self.study),
                **_hs_tile_meta(tile, self.study),
            }
            for slide in self.study["slides"].values()
            for tile in slide["tiles"].values()
        ]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.tiles):
            self.__iter__()
            raise StopIteration
        else:
            if (
                self.large_images is None
            ):  # lazy creation of large_image objects for serialization
                slides = set([tile[0] for tile in self.tiles])
                self.large_images = {slide: large_image.open(slide) for slide in slides}
            indices = range(self.i, min(self.i + max(self.batch, 1), len(self.tiles)))
            pixels = [
                self.large_images[self.tiles[j][0]].getRegion(**self.tiles[j][1])[0]
                for j in indices
            ]
            metadata = {
                k: np.array([self.metadata[j][k] for j in indices])
                for k in self.metadata[indices[0]]
            }
            if self._singleton:
                pixels = pixels[0]
            else:
                pixels = np.stack(pixels, axis=0)
            if self._nchw:
                if self._singleton:
                    axes = [2, 0, 1]
                else:
                    axes = [0, 3, 2, 1]
                pixels = np.transpose(pixels, axes)
            self.i += len(indices)
            if (self.w_source is None) and (self.w_target is None):
                return pixels, metadata
            else:
                data = dict(
                    input_0=pixels, input_1=self.w_source, input_2=self.w_target
                )
                return data, metadata
