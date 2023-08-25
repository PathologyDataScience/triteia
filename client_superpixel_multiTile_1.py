##############################################################################
# Superpixel Segmentation with SLIC using Triton Inference Server and Python backend
# Author: Ahmad
# Date: 2023-08-03
# Description: This script uses the Triton Inference Server to apply the SLIC
#              (Simple Linear Iterative Clustering) superpixel segmentation
#              algorithm on an input whole slide image (.svs). The Triton
#              server runs a Python backend model that leverages the SLIC
#              implementation from the scikit-image library. The resulting
#              superpixel segmented image is returned as the output.
##############################################################################

import functools
import os
import numpy as np
import tensorflow as tf
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from mil.io.utils import study
from simple_triton.feature_extraction import histomics_stream_inference
from skimage.transform import resize
from skimage.segmentation import slic
import openslide
import time
from large_image.cache_util import cachesClear



def warmup_model():
    # Create Triton HTTP client
    triton_client_grpc = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)



    # Create grpc InferenceRequest
    # Create grpc InferenceRequest
    inputs0 = []
    inputs0.append(grpcclient.InferInput("INPUT0", (1, 4096, 4096, 3), "FP32"))  # Add batch dimension

    # Iterate through all the tiles
    x = 0  # Tile index for the x direction
    y = 0  # Tile index for the y direction
    x_start, y_start = x * tile_size[0], y * tile_size[1]
    x_end, y_end = x_start + tile_size[0], y_start + tile_size[1]

    tile = slide.read_region((x_start, y_start), 0, (x_end - x_start, y_end - y_start))
    tile = np.array(tile)[:, :, :3]  # Convert RGBA to RGB
    input_data_0 = np.array(tile, dtype=np.float32)

    # Reshape the input data to include the batch dimension
    input_data_reshaped = input_data_0[np.newaxis, ...]
    inputs0[0].set_data_from_numpy(input_data_reshaped)

    outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]

    # Measure the time taken for inference with Triton
    start_time = time.time()
    response = triton_client_grpc.infer(model_name, inputs=inputs0, outputs=outputs)
    end_time = time.time()
    time_elapsed_triton = end_time - start_time

    print("Time Elapsed for Inference with Triton: {:.8f} seconds".format(time_elapsed_triton))
def client_nogpu():
    """Run client with no GPUs"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    assert len(tf.config.list_physical_devices("GPU")) == 0

def cache_clear():
    """clearn the cache before running inference"""
    cachesClear()

    @functools.lru_cache(maxsize=None)
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    def gfg():
        fib.cache_clear()

    fib(30)
    # Before Clearing
    print(fib.cache_info())
    gfg()
    # After Clearing
    print(fib.cache_info())

def gpu_mem_clear():
    """clearn the memory before running inference"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


from simple_triton.feature_extraction import histomics_stream_inference


def create_hs_study(wsi_path, tile, chunk, target_magnification=20, source="exact"):
    # ...

    """Create a histomic stream study.

    Parameters used in this cell are for reading
    from the whole-slide image (magnification, tile size, tile overlap, mask file).

    Args:
        args_dict (dict): The inputs to the model from argparse.
        tile (int): tile default value is 224.
        
        wsi_path (string): path for .svs file
        
        mask_path (string): path for png file
        
        chunk : tuple(int, int)
            The size of a region to retrieve from the slide at the target magnification. histomics_stream
            groups the reading of tiles into multi-tile chunks to minimize overhead. Default value is
            (1792, 1792).
        target : float
            The target magnification to return tiles at. If this magnification is not available, histomics_stream
            will read the next highest magnification and resize to obtain the desired magnification. Default
            value is 20 to analyze at 20X objective magnification.
        source : string
            A histomics_stream parameter defining read and resizing behavior. Default value exact returns the
            exact magnification requested, using resizing if necessary. See histomics_stream documentation
            for more details.
    """
    # slide     parameters
    
    # create a histomic-stream study from a wsi/mask pair
    hs_study = study(
        (wsi_path),
        t=(tile, tile),
        chunk=(tile, tile),
        target=target_magnification,
        source="exact",
    )


    return hs_study

def custom_histomics_stream_inference(hs_study, model_name, url, batch, workers, limit):

    return histomics_stream_inference(hs_study, model_name, url=url, batch=batch, workers=workers, limit=limit)
   
model_name = "superpixel_slic"
# model_name = "superpixel_slic_skimage"
# Set up slide and tile parameters
slide_path = ["/tf/notebook/TCGA-TM-A7CF-01Z-00-DX1.EB905EDB-5AC5-41C4-AB00-526DD3820524.svs"]
slide_path_str = ', '.join(slide_path)
tile_size = (4096, 4096)
tile = 4096
num_tiles = 0


slide = openslide.OpenSlide(slide_path_str)

magnification = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
(width, height) = slide.dimensions
factors = slide.level_downsamples

# Display metadata
print("slide scanned at {} objective magnification".format(magnification))
print("scanned image is {0} x {1} pixels".format(width, height))
print("contains {0} levels with downsample factors: {1}".format(len(factors), factors))

total_tiles_width = width // tile_size[0]
total_tiles_height = height // tile_size[1]
total_tiles = total_tiles_width * total_tiles_height
num_tiles = total_tiles
print("Total number of tiles: {0}, Width: {1}, Height: {2}".format(total_tiles, total_tiles_width, total_tiles_height))


# Create Triton HTTP client
triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
triton_client_grpc = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

# client_nogpu()
# cache_clear()
# gpu_mem_clear()


# Load model
try:
    triton_client.unload_model(model_name)
    triton_client.load_model(model_name)
    print(f"Model '{model_name}' loaded successfully.")
except Exception as e:
    print(f"Failed to load model '{model_name}': {e}")


################################################################################

# warmup_model()

################################################################################

# Create a histomic-stream study
print("Create a histomic-stream study")
hs_study = create_hs_study(slide_path, tile, tile, 20, "exact")


outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]

# Perform custom histomics stream inference
print("Perform custom histomics stream inference")
url="localhost:8001",
batch=1,
workers=32,
limit=10


num_iterations = 1
average_throughput_tiles_cuda = 0

for _ in range(num_iterations):
    start_time = time.time()
    response, tile_info, times, failed = histomics_stream_inference(hs_study, model_name, url="localhost:8001", batch=1, workers=32, limit=10)
    print("response",response)
    print("response",outputs)
    
    elapsed_time = time.time() - start_time
    throughput_tiles_cuda = total_tiles / elapsed_time
    average_throughput_tiles_cuda += throughput_tiles_cuda

average_throughput_tiles_cuda /= num_iterations

print("            ----------------------------------            ")
print("Average Throughput for CUDA-SLIC or Skimage SLIC in terms of tiles/sec:", average_throughput_tiles_cuda, "tiles/second")
print("Average elapsed_time for CUDA-SLIC or Skimage SLIC in terms of seconds:", elapsed_time, "seconds")

print("FINISHED Inference")
