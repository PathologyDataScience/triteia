from copy import deepcopy
import large_image_source_tiff as large_image
import numpy as np


def _byteify(string):
    return bytes(string.encode('utf-8'))

def _hs_study_meta(study):
    return {
        "version": _byteify(study["version"]),
        "tile_height": study["tile_height"],
        "tile_width": study["tile_width"],
        "overlap_height": study["overlap_height"],
        "overlap_width": study["overlap_width"],
    }


def _hs_slide_meta(slide, study):
    return {
        "filename": _byteify(slide["filename"]),
        "slide_name": _byteify(slide["slide_name"]),
        "slide_group": _byteify(slide["slide_name"]),
        "target_magnification": slide["target_magnification"],
        "scan_magnification": slide["target_magnification"],
        "read_magnification": slide["read_magnification"],
        "returned_magnification": slide["returned_magnification"],
        "level": slide["level"],
        "slide_width": slide["slide_width"],
        "slide_height": slide["slide_height"],
        "chunk_width": study["tile_width"],
        "chunk_height": study["tile_height"],
        "slide_height_tiles": slide["slide_height_tiles"],
        "slide_width_tiles": slide["slide_width_tiles"],
        "mask_height": slide["mask_height"],
        "mask_width": slide["mask_width"],
    }


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
    batch : int
        The number of tiles in each batch. Partial batches are not padded.
    worker_index : int
        The worker index, ranging from 0 to `num_workers`.
    num_workers : int
        The total number of workers.

    Returns
    -------
    tiles : array-like
        A four-dimensional BHWC numpy array of batched tiles.
    metadata : dict
        TBD

    Attributes
    ----------
    batch : int
        The number of tiles in each batch. Partial batches are not padded.
    worker_index : int
        The worker index, ranging from 0 to `num_workers`.
    num_workers : int
        The total number of workers.

    Notes
    -----
    See https://github.com/DigitalSlideArchive/HistomicsStream/blob/master/StudyObject.md
    for more details on `metadata`.

    """

    def __init__(self, study, batch, worker_index, num_workers):
        self.i = 0
        self.study = study
        self.batch = batch
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.large_images = None
        self._shard()

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
                self.large_images = {
                    self.tiles[self.i][0]: large_image.open(self.tiles[self.i][0])
                }
            else:
                if (
                    self.tiles[self.i][0] not in self.large_images
                ):  # shards can span multiple slides - slide not open yet
                    self.large_images[self.tiles[self.i][0]] = large_image.open(
                        self.tiles[self.i][0]
                    )
            indices = range(self.i, min(self.i + self.batch, len(self.tiles)))
            pixels = [
                self.large_images[self.tiles[j][0]].getRegion(**self.tiles[j][1])[0]
                for j in indices
            ]
            metadata = {
                k: np.array([self.metadata[j][k] for j in indices])
                for k in self.metadata[indices[0]]
            }
            pixels = np.stack(pixels, axis=0)
            self.i += len(indices)
            return pixels, metadata
