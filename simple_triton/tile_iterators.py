from collections import deque
from concurrent.futures import ProcessPoolExecutor, ALL_COMPLETED, wait
import functools
import histomics_stream as hs
import large_image_source_tiff
import math
import multiprocessing.shared_memory
import numpy as np
import operator
import os
import PIL


class SharedNumpyArray:
    def __init__(self, shape, dtype):
        """Init"""
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.shm_size = functools.reduce(operator.mul, shape, 1) * self.dtype.itemsize
        self.shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=self.shm_size
        )
        self.buf = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.created = True

    def insert(self, arr, i):
        """Insert a batch dimension slice."""
        self.buf[i] = arr

    def copy(self, arr):
        self.shape = arry.shape
        self.buf = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.buf[:] = arr[:]

    def tobytes(self):
        return self.buf.tobytes()

    def view(self):
        return np.ndarray(self.shape, self.dtype, buffer=self.shm.buf)

    # If we want easier interoperability, we could, instead, forward a
    # whitelist of attributes to our underlying np.ndarray object; these could
    # be enumerated and done via __getattribute__
    def __getitem__(self, idx):
        return self.buf[idx]

    def __array__(self, dtype=None):
        return self.buf.copy().astype(dtype) if dtype is not None else self.buf.copy()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["shm"]
        state.pop("buf", None)
        state["created"] = False
        state["shm_name"] = self.shm.name
        return state

    def __setstate__(self, state):
        state = state.copy()
        shm_name = state.pop("shm_name")
        self.__dict__.update(state)
        self.shm = multiprocessing.shared_memory.SharedMemory(shm_name)
        self.buf = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def __del__(self):
        if hasattr(self, "shm"):
            self.shm.close()
            if getattr(self, "created", None) is True:
                self.shm.unlink()


def _txr_keys(dictionary, keys):
    return {
        k: (
            dictionary[k].encode("utf-8")
            if isinstance(dictionary[k], str)
            else dictionary[k]
        )
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


def _hs_tile_meta(tile, chunk, study):
    return {
        "chunk_top": chunk["chunk_top"],
        "chunk_left": chunk["chunk_left"],
        "chunk_bottom": chunk["chunk_bottom"],
        "chunk_right": chunk["chunk_right"],
        "tile_top": tile["tile_top"],
        "tile_left": tile["tile_left"],
    }


def _hs_flatten(slide, study):
    """Flattens a histomics stream study into a list of tiles, or a list of
    lists of tiles (when chunking)."""

    if "chunks" in slide.keys():
        flattened = [
            [
                {
                    **_hs_study_meta(study),
                    **_hs_slide_meta(slide, study),
                    **_hs_tile_meta(tile, chunk, study),
                }
                for tile in chunk["tiles"].values()
            ]
            for chunk in slide["chunks"].values()
        ]
    else:
        flattened = [
            {
                **_hs_study_meta(study),
                **_hs_slide_meta(slide, study),
                **_hs_tile_meta(
                    tile,
                    {
                        "chunk_top": tile["tile_top"],
                        "chunk_left": tile["tile_left"],
                        "chunk_bottom": tile["tile_top"] + study["tile_height"],
                        "chunk_right": tile["tile_left"] + study["tile_width"],
                    },
                    study,
                ),
            }
            for slide in study["slides"].values()
            for tile in slide["tiles"].values()
        ]
    return flattened


class TiffPrefetch(object):
    """An eager tile iterator for large_image_tiff formats.

    This iterator generates tile batches via multiprocessing. Each process
    reads a group of tiles to minimize reads.

    Parameters
    ----------
    study : dict
        A histomics stream study.
    dtype : type
        Desired numpy datatype for outputs. Default value is `np.uint8`.
    icc : bool
        Whether to attempt ICC correction.
    batch : int
        The batch size. A partial batch at the end will not be padded.
    prefetch : int
        The target number of prefetched batches.
    workers : int
        The number of multiprocessing workers.
    """

    def __init__(
        self, study, dtype=np.uint8, icc=False, batch=64, prefetch=16, workers=16
    ):
        if len(study["slides"]) > 1:
            raise ValueError("Multi-slide studies not supported.")
        self.dtype = dtype
        self.pool = ProcessPoolExecutor(max_workers=workers)
        slide = list(study["slides"].values())[0]
        self.source = large_image_source_tiff.open(
            slide["filename"], style={"icc": False} if not icc else None
        )
        self.read_kwargs = _hs_flatten(slide, study)
        self._initialize(batch, prefetch)

    def _initialize(self, batch, prefetch):
        self.prefetch = prefetch
        self.batch = batch
        self.queue = deque([])  # hold futures defining read operations
        self.overflow = 0  # count of tile overrun for latest batch
        self.pos = 0  # position in read_kwargs
        self._fill()

    def __iter__(self):
        return self

    def __next__(self):
        if (self.pos == len(self.read_kwargs)) and not len(self.queue):
            raise StopIteration

        # wait on the futures linked to the next batch
        try:
            futures, tiles, read_kwargs = self.queue.pop()
            wait(futures, timeout=None, return_when=ALL_COMPLETED)
            self._fill()
        except:
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise

        # last batch may only have partial size
        if self.pos == len(self.read_kwargs) and not len(self.queue):
            tiles.shape = [len(read_kwargs), *tiles.shape[1:]]

        return tiles, read_kwargs

    @staticmethod
    def read(source, dtype, read_kwargs, sharrs, offset, batch):
        # read followed by crops
        xt = [k["tile_left"] for k in read_kwargs]
        yt = [k["tile_top"] for k in read_kwargs]
        wt = [k["tile_width"] for k in read_kwargs]
        ht = [k["tile_height"] for k in read_kwargs]
        xr = min(xt)
        yr = min(yt)
        wr = max([x + w for (x, w) in zip(xt, wt)]) - xr
        hr = max([y + h for (y, h) in zip(yt, ht)]) - yr
        kwargs = dict(
            scale={"magnification": read_kwargs[0]["target_magnification"]},
            format="numpy",
            region=dict(left=xr, top=yr, width=wr, height=hr, units="mag_pixels"),
            tile_size=dict(width=wr, height=hr),
        )
        chunk, _ = source.getRegion(**kwargs)
        tiles = [
            chunk[y - yr : y - yr + h, x - xr : x - xr + w, :].astype(dtype)
            for (x, y, w, h) in zip(xt, yt, wt, ht)
        ]
        for i, tile in enumerate(tiles):
            sharr_index, slice_index = divmod(offset + i, batch)
            sharrs[sharr_index].insert(tile, slice_index)

    def _submitfn(self, read_kwargs, sharrs, offset):
        return self.pool.submit(
            self.read,
            self.source,
            self.dtype,
            read_kwargs,
            sharrs,
            offset,
            self.batch,
        )

    def _fill(self):
        try:
            while len(self.queue) < self.prefetch and self.pos < len(self.read_kwargs):
                """if the last read from the prior batch spanned batch boundaries then
                the leftmost element in self.queue contains the futures, shared array,
                and read_kwargs to start the current batch"""
                if self.overflow:
                    futures, tiles, batch_kwargs = self.queue.popleft()
                else:
                    """last read aligned with batch boundary, create new shared array,
                    and read_kwargs, futures containers"""
                    tiles = SharedNumpyArray(
                        [
                            self.batch,
                            self.read_kwargs[self.pos][0]["tile_height"],
                            self.read_kwargs[self.pos][0]["tile_width"],
                            3,
                        ],
                        self.dtype,
                    )
                    futures = []
                    batch_kwargs = []

                """submit enough jobs to fill at least one batch - a single job may 
                fill multiple batches - or multiple jobs may be needed to fill one 
                batch"""
                offset = self.overflow
                batches = 1
                tiles = [tiles]
                while len(batch_kwargs) < self.batch and self.pos < len(
                    self.read_kwargs
                ):
                    # number of batches spanned by this read
                    reads = self.read_kwargs[self.pos]
                    batches = math.ceil((len(reads) + offset) / self.batch)

                    # create additional arrays if this read spans multiple batches
                    tiles = [
                        *tiles,
                        *[
                            SharedNumpyArray(
                                [
                                    self.batch,
                                    reads[0]["tile_height"],
                                    reads[0]["tile_width"],
                                    3,
                                ],
                                self.dtype,
                            )
                            for _ in range(batches - 1)
                        ],
                    ]

                    """submit job - read into first array in `tiles` at slice `offset`.
                    overflow to subsequent arrays if this read spans multiple batches.
                    if multiple reads are required to fill this batch then increment
                    the offset and submit another job on the next iteration"""
                    futures.append(self._submitfn(reads, tiles, offset))
                    offset = offset + len(reads)
                    batch_kwargs = batch_kwargs + reads
                    self.pos = self.pos + 1
                self.overflow = len(batch_kwargs) % self.batch

                """ if last read spans multipe batches, link that read's future
                to the other batches - also divide kwargs according to batch boundaries
                """
                futures = [futures] + (batches - 1) * [[futures[-1]]]
                batch_kwargs = [
                    batch_kwargs[i : i + self.batch]
                    for i in range(0, len(batch_kwargs), self.batch)
                ]

                # enqueue batches
                for f, t, b in zip(futures, tiles, batch_kwargs):
                    self.queue.appendleft((f, t, b))

        except Exception as error:
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise

