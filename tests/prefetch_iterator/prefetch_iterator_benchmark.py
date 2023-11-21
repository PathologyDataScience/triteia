from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import large_image
import large_image_source_tiff
from large_image.cache_util import cachesClear
from queue import Queue
from time import sleep, time
from mil.io.utils import study
import os
import pooch
# multi processing and threading

# Class for prefetching image regions from different sources
class PrefetchIterator(object):
    def __init__(self, file, read_kwargs, workers, reader):
        # Choose the appropriate source based on the specified reader
        if reader == "open_source":
            self.source = large_image.open(file)
        elif reader == "tiff":
            self.source = large_image_source_tiff.open(file)
        self.read_kwargs = read_kwargs
        self.pool = ProcessPoolExecutor(max_workers=workers)
        self.queue = Queue(workers)
        self.i = 0
        self._fill()

    def __iter__(self):
        # Initialize the iterator
        self.i = 0
        return self

    def __next__(self):
        # Get the next prefetched image region
        if self.i >= len(self.read_kwargs):
            raise StopIteration
        future = self.queue.get()
        wait([future], timeout=None, return_when=FIRST_COMPLETED)
        self.i += 1
        try:
            item = future.result()
        except Exception as error:
            print(error)
            self.pool.shutdown(wait=False, cancel_futures=True)
            # raise exc
        self._fill()
        return item

    def _fill(self):
        while not self.queue.full() and self.i + self.queue.qsize() < len(
            self.read_kwargs
        ):
            self.queue.put(
                self.pool.submit(
                    self.source.getRegion,
                    **self.read_kwargs[self.i + self.queue.qsize()],
                )
            )


# download whole slide image and corresponding mask
wsi_path = pooch.retrieve(
    fname="TCGA-AN-A0G0-01Z-00-DX1.svs",
    url="https://drive.google.com/uc?export=download&id=19agE_0cWY582szhOVxp9h3kozRfB4CvV&confirm=t&uuid=6f2d51e7-9366-4e98-abc7-4f77427dd02c&at=ALgDtswlqJJw1KU7P3Z1tZNcE01I:1679111148632",
    known_hash="d046f952759ff6987374786768fc588740eef1e54e4e295a684f3bd356c8528f",
    path=str(pooch.os_cache("pooch")) + os.sep + "wsi",
)
mask_path = pooch.retrieve(
    fname="TCGA-AN-A0G0-01Z-00-DX1.mask.png",
    url="https://drive.google.com/uc?export=download&id=17GOOHbL8Bo3933rdIui82akr7stbRfta",
    known_hash="bb657ead9fd3b8284db6ecc1ca8a1efa57a0e9fd73d2ea63ce6053fbd3d65171",
    path=str(pooch.os_cache("pooch")) + os.sep + "wsi",
)


def get_read_kwargs(hs_study):
    # Generate a list of keyword arguments for reading image regions based on the study
    read_kwargs = [
        {
            "scale": {"magnification": slide["target_magnification"]},
            "format": "numpy",
            "region": {
                "left": tile["tile_left"],
                "top": tile["tile_top"],
                "width": hs_study["tile_width"],
                "height": hs_study["tile_height"],
                "units": "mag_pixels",
            },
        }
        for slide in hs_study["slides"].values()
        for tile in slide["tiles"].values()
    ]
    return read_kwargs


def run_naive_benchmark(tile, magnification, overlap, read_kwargs, reader):
    # Run the naive benchmark for reading image regions without prefetching
    source = large_image_source_tiff.open(wsi_path)
    if reader == "open_source":
        source = large_image.open(wsi_path)
    elif reader == "tiff":
        source = large_image_source_tiff.open(wsi_path)
    # Clear the large_image cache
    cachesClear()
    try:
        start = time()
        for i, kwargs in enumerate(read_kwargs):
            tile, _ = source.getRegion(**kwargs)
            tile[0, 0, 0]  # # In case of lazy reading, access a pixel to force reading
        average_iteration_time = (time() - start) / i
        print(
            f"Naive_average_iteration_time for {reader}: {average_iteration_time:.5f} seconds"
        )
        file1 = open("prefetch_iterator_benchmark_opensource_versus_tiff.txt", "a+")
        file1.write(
            f"\nTile_Size {tile_size} Magnification {magnification} Workers {workers} overlap {overlap}"
        )
        file1.write(f" Naive_average_iteration_time(sec) {average_iteration_time:.5f}")
        file1.close()
    except Exception as e:
        print("An exception occurred ", str(e))


def run_benchmark(tile, magnification, workers, overlap, read_kwargs, reader):
    # Run the benchmark using the PrefetchIterator for parallel reading
    # Clear the large_image cache
    cachesClear()
    try:
        # read the image using a prefetch iterator
        start = time()
        it = PrefetchIterator(wsi_path, read_kwargs, workers=workers, reader=reader)
        for i in range(len(read_kwargs)):
            tile, _ = next(it)
            tile[0, 0, 0]  # # In case of lazy reading, access a pixel to force reading
        average_iteration_time = (time() - start) / len(read_kwargs)
        print(
            f"Prefetch average iteration time for {reader}: {average_iteration_time:.5f} seconds"
        )
        file1 = open("prefetch_iterator_benchmark_opensource_versus_tiff.txt", "a+")
        file1.write(
            f"\nTile_Size {tile_size} Magnification {magnification} Workers {workers} overlap {overlap}"
        )
        file1.write(
            f" prefetch_average_iteration_time_{reader} {average_iteration_time:.5f}"
        )
        file1.close()
    except Exception as e:
        print("An exception occurred ", str(e))


# Benchmark different values
o = 1  # for overlap
magnifications = [5, 10, 20, 40]  # maginification:[5,10,20,40]
tile_sizes = [224, 448, 672, 896, 1120]  # Tile sizes:[224,448,672,896,1120]
worker_counts = [1, 2, 4, 8, 16, 32, 64]  # Number of worker:[1,2,4,8,16,32,64]
overlap_counts = [0, o]

for magnification in magnifications:
    for tile_size in tile_sizes:
        for overlap in overlap_counts:
            if overlap == 1:
                overlap = int(tile_size / 2)
            for workers in worker_counts:
                # create a histomics-stream study from a wsi/mask pair
                hs_study = study(
                    (wsi_path, mask_path),
                    t=(tile_size, tile_size),
                    overlap=(overlap, overlap),
                    chunk=(tile_size, tile_size),
                    target=magnification,
                    source="exact",
                )
                # transform study to a list of kwargs for getRegion
                read_kwargs = get_read_kwargs(hs_study)
                print(
                    f"\nBenchmarking: Tile Size={tile_size}, Magnification={magnification}, Workers={workers}, overlap={overlap}"
                )
                run_naive_benchmark(
                    tile_size, magnification, overlap, read_kwargs, reader="open_source"
                )
                run_benchmark(
                    tile_size,
                    magnification,
                    workers,
                    overlap,
                    read_kwargs,
                    reader="open_source",
                )
                run_naive_benchmark(
                    tile_size, magnification, overlap, read_kwargs, reader="tiff"
                )
                run_benchmark(
                    tile_size,
                    magnification,
                    workers,
                    overlap,
                    read_kwargs,
                    reader="tiff",
                )
