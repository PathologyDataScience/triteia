from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    ALL_COMPLETED,
    FIRST_COMPLETED,
    wait,
)
from copy import deepcopy
import functools
import histomics_stream as hs
import large_image_source_tiff
import multiprocessing.shared_memory
import numpy as np
import operator
import os
import PIL
from queue import Queue


class SharedNumpyArray:
    def __init__(self, shape, dtype):
        """Init"""
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.shm_size = functools.reduce(operator.mul, shape, 1) * self.dtype.itemsize
        self.shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=self.shm_size
        )
        self.created = True

    def copy(self, arr):
        self.shape = arr.shape
        self.buf = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.buf[:] = arr[:]

    def tobytes(self):
        return self.buf.tobytes()

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
                    **_hs_study_meta(slide),
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


def _largeimage_kwargs(slide, study):
    """Transform histomics_stream tile definitions to read kwargs for
    use with large_image. Return read keyword arguments and corresponding
    study values."""

    def tile_dict(tile, slide, study):
        return dict(
            scale={"magnification": slide["target_magnification"]},
            format="numpy",
            region=dict(
                left=tile["tile_left"],
                top=tile["tile_top"],
                width=study["tile_width"],
                height=study["tile_height"],
                units="mag_pixels",
            ),
            tile_size=dict(width=study["tile_width"], height=study["tile_height"]),
        )

    flattened = _hs_flatten(slide, study)
    if "chunks" in slide.keys():
        read_kwargs = [[tile_dict(t, slide, study) for t in c] for c in flattened]
    else:
        read_kwargs = [tile_dict(t, slide, study) for t in flattened]

    return read_kwargs, flattened


class LargeimagePrefetch(object):
    """A prefetching tile iterator for large_image tiff sources.

    This iterator generates tile batches via multiprocessing. If chunking
    is specified in input study, each process will read one chunk. Otherwise
    each process reads one tile.

    Parameters
    ----------
    study : dict
        A histomics stream study.
    icc : bool
        Whether to apply ICC correction. Default value is True.
    batch : int
        Size of generated batches. Partial batches are not padded. Default 
        value is 64.
    prefetch : int
        The number of prefetched batches to maintain.
    workers : int
        The number of multiprocessing workers.
    """

    def __init__(self, study, icc=True, batch=64, prefetch=4, workers=32):
        if len(study["slides"]) > 1:
            raise ValueError("Multi-slide studies not supported.")
        slide = list(study["slides"].values())[0]
        self.pool = ProcessPoolExecutor(max_workers=workers)
        self.source = large_image_source_tiff.open(
            slide["filename"], style={"icc": False} if not icc else None
        )
        self.read_kwargs, self.meta = _largeimage_kwargs(slide, study)
        self.group = "chunks" in slide.keys()
        self.prefetch = prefetch
        self.batch = batch
        self.pos = 0
        if self.group:
            self._initialize_group(batch, prefetch)
        else:
            self._initialize_single(batch, prefetch)

    def _initialize_single(self, batch, prefetch):
        self.queue = Queue(prefetch)
        self.pos = 0  # tracks number of read tiles
        self._fill_single()

    def _initialize_group(self, batch, prefetch):
        self.queue = Queue()  # hold futures defining read operations
        self.pending = 0  # number of pending tiles in futures
        self.remainder = ([], [])  # save overflow for following batch
        self._fill_group()

    def __iter__(self):
        return self

    def __next__(self):
        if self.group:
            return self._next_group()
        else:
            return self._next_single()

    def _next_group(self):
        if (
            (self.pos == len(self.read_kwargs))
            and (self.pending == 0)
            and (len(self.remainder[0]) == 0)
        ):
            raise StopIteration

        # add partial batch remainder samples
        tiles, meta = self.remainder
        self.remainder = ([], [])

        # fill to full batch or exhaustion
        while (len(tiles) < self.batch) and (self.pending > 0):
            future, m = self.queue.get()
            wait([future], timeout=None, return_when=FIRST_COMPLETED)
            try:
                reads = future.result()
            except Exception as error:
                self.pool.shutdown(wait=False, cancel_futures=True)
                raise
            if len(reads) + len(tiles) > self.batch:
                self.remainder = (
                    reads[self.batch - len(tiles) :],
                    m[self.batch - len(tiles) :],
                )
            meta = meta + m[0 : min(len(reads), self.batch - len(tiles))]
            tiles = tiles + reads[0 : min(len(reads), self.batch - len(tiles))]
            self.pending = self.pending - len(reads)
        self._fill_group()
        return tiles, meta

    @staticmethod
    def read_group(source, sharr, read_kwargs):
        # read followed by crops
        xt = [k["region"]["left"] for k in read_kwargs]
        yt = [k["region"]["top"] for k in read_kwargs]
        wt = [k["region"]["width"] for k in read_kwargs]
        ht = [k["region"]["height"] for k in read_kwargs]
        xr = min(xt)
        yr = min(yt)
        wr = max([x + w for (x, w) in zip(xt, wt)]) - xr
        hr = max([y + h for (y, h) in zip(yt, ht)]) - yr
        read_dict = deepcopy(read_kwargs[0])
        read_dict["region"]["left"] = xr
        read_dict["region"]["top"] = yr
        read_dict["region"]["width"] = wr
        read_dict["region"]["height"] = hr
        read_dict["tile_size"] = {"width": wr, "height": hr}
        chunk, _ = source.getRegion(**read_dict)
        for s, x, y, w, h in zip(sharr, xt, yt, wt, ht):
            s.copy(chunk[y - yr : y - yr + h, x - xr : x - xr + w, :])
        return sharr

    def _submit_group(self):
        return self.pool.submit(
            self.read_group,
            self.source,
            [
                SharedNumpyArray(
                    (r["region"]["height"], r["region"]["width"], 3), np.uint8
                )
                for r in self.read_kwargs[self.pos]
            ],
            self.read_kwargs[self.pos],
        )

    def _fill_group(self):
        try:
            while len(
                self.remainder[0]
            ) + self.pending < self.batch * self.prefetch and self.pos < len(
                self.read_kwargs
            ):
                self.queue.put((self._submit_group(), self.meta[self.pos]))
                self.pending = self.pending + len(self.read_kwargs[self.pos])
                self.pos = self.pos + 1
        except Exception as error:
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise

    def _next_single(self):
        if self.pos >= len(self.read_kwargs):
            raise StopIteration
        futures = self.queue.get()
        wait(futures, timeout=None, return_when=ALL_COMPLETED)
        self.pos += len(futures)
        try:
            batch = [(f.result(), m) for (f, m) in futures.items()]
            tiles = [b[0] for b in batch]
            meta = [b[1] for b in batch]
        except Exception as error:
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise
        self._fill_single()
        return tiles, meta

    @staticmethod
    def read_single(source, sharr, **kwargs):
        tile, _ = source.getRegion(**kwargs)
        sharr.copy(tile)
        return sharr

    def _submit_single(self, *args, **kwargs):
        return self.pool.submit(
            self.read_single,
            self.source,
            SharedNumpyArray(
                (kwargs["region"]["height"], kwargs["region"]["width"], 3), np.uint8
            ),
            **kwargs
        )

    def _fill_single(self):
        try:
            start = self.pos + self.batch * self.queue.qsize()
            while not self.queue.full() and start < len(self.read_kwargs):
                futures = {}
                for k, m in zip(
                    self.read_kwargs[
                        start : min(start + self.batch, len(self.read_kwargs))
                    ],
                    self.meta[start : min(start + self.batch, len(self.meta))],
                ):
                    futures[self._submit_single(**k)] = m
                self.queue.put(futures)
                start += self.batch  # Move to the next batch
        except Exception as error:
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise
