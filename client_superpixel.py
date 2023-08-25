# import numpy as np
# import tritonclient.http as httpclient
# from skimage.io import imread

# model_name = "skimage_slic"
# input_image_path = '/tf/notebook/TCGA-05-4425-01Z-00-DX1.82B093EE-49BC-4FD9-91AC-4CC89944309D.svs'
# wsi = imread(input_image_path)
# n_segments = int(wsi.shape[0] * wsi.shape[1] / 125)

# # Create Triton HTTP client
# triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)
# model_name = "superpixel_slic"
# # Load model
# try:
#     triton_client.load_model(model_name)
#     print(f"Model '{model_name}' loaded successfully.")
# except Exception as e:
#     print(f"Failed to load model '{model_name}': {e}")

# # Create InferenceRequest
# input_data = np.array([n_segments], dtype=np.int32)

# inputs = []
# inputs.append(httpclient.InferInput("INPUT0", input_data.shape, "INT32"))
# inputs[0].set_data_from_numpy(input_data)

# outputs = [
#     httpclient.InferRequestedOutput("OUTPUT0"),
# ]

# response = triton_client.infer(model_name,
#                                inputs,
#                                request_id=str(1),
#                                outputs=outputs)

# # Process and print the output
# output_0 = response.as_numpy("OUTPUT0")
# print("Number of segments:", output_0)

#Latest working code July 27 2023 9:15 AM
# import numpy as np
# import tritonclient.http as httpclient
# from skimage.io import imread

# model_name = "superpixel_slic"

# # Read the whole slide image file (.svs) using skimage
# input_image_path = '/tf/notebook/TCGA-05-4425-01Z-00-DX1.82B093EE-49BC-4FD9-91AC-4CC89944309D.svs'
# wsi = imread(input_image_path)

# # Calculate the value of 'n_segments' for SLIC segmentation
# n_segments =  4096 # int(wsi.shape[0] * wsi.shape[1] / 125)
# print("n_segments:")
# print(n_segments)
# # Convert 'n_segments' to a list (as the model expects a list)
# input_data_0 = [n_segments]

# # Create Triton HTTP client
# triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)

# # Load model
# try:
#     triton_client.load_model(model_name)
#     print(f"Model '{model_name}' loaded successfully.")
# except Exception as e:
#     print(f"Failed to load model '{model_name}': {e}")

# # Create InferenceRequest
# inputs = []
# inputs.append(httpclient.InferInput("INPUT0", [len(input_data_0)], "FP32"))
# inputs[0].set_data_from_numpy(np.array(input_data_0, dtype=np.float32))

# outputs = [
#     httpclient.InferRequestedOutput("OUTPUT0"),
# ]

# response = triton_client.infer(model_name,
#                                inputs,
#                                request_id=str(1),
#                                outputs=outputs,
#                                timeout=1800000 
#                                 )


# # Process and print the output
# output_0 = response.as_numpy("OUTPUT0")
# print("Input n_segments:", input_data_0)
# print("Output segmented image shape:", output_0.shape)



# import numpy as np
# import tritonclient.http as httpclient
# from skimage.io import imread

# model_name = "superpixel_slic"

# # Read the whole slide image file (.svs) using skimage
# input_image_path = '/tf/notebook/TCGA-05-4425-01Z-00-DX1.82B093EE-49BC-4FD9-91AC-4CC89944309D.svs'
# wsi = imread(input_image_path)

# # Convert image data to a list (assuming the model expects a list)
# input_data_0 = wsi.tolist()

# # Create Triton HTTP client
# triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)

# # Load model
# try:
#     triton_client.load_model(model_name)
#     print(f"Model '{model_name}' loaded successfully.")
# except Exception as e:
#     print(f"Failed to load model '{model_name}': {e}")

# # Create InferenceRequest
# inputs = []
# inputs.append(httpclient.InferInput("INPUT0", [len(input_data_0)], "FP32"))
# inputs[0].set_data_from_numpy(np.array(input_data_0, dtype=np.float32))

# outputs = [
#     httpclient.InferRequestedOutput("OUTPUT0"),
# ]

# response = triton_client.infer(model_name,
#                                inputs,
#                                request_id=str(1),
#                                outputs=outputs,
#                                timeoout = 1800000
#                                 )


# # Process and print the output
# output_0 = response.as_numpy("OUTPUT0")
# print("Input image shape:", wsi.shape)
# print("Output segmented image shape:", output_0.shape)



# # Latest working code July, 27 2023. Time: 10:07 AM
# import numpy as np
# import tritonclient.http as httpclient
# from skimage.io import imread
# from skimage.segmentation import slic

# model_name = "superpixel_slic"

# # Read the whole slide image file (.svs) using skimage
# input_image_path = '/tf/notebook/TCGA-05-4425-01Z-00-DX1.82B093EE-49BC-4FD9-91AC-4CC89944309D.svs'
# wsi = imread(input_image_path)

# # Desired number of superpixels (n_segments)
# n_segments = 4096

# # Perform SLIC segmentation on the input image
# segments_slic = slic(wsi, n_segments=n_segments, compactness=10, sigma=1)

# # Convert the segmented image to a 1D numpy array with datatype INT32
# input_data_0 = np.array([n_segments], dtype=np.float32)

# # Create Triton HTTP client
# triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)

# # Load model
# try:
#     triton_client.load_model(model_name)
#     print(f"Model '{model_name}' loaded successfully.")
# except Exception as e:
#     print(f"Failed to load model '{model_name}': {e}")

# # Create InferenceRequest
# inputs = []
# inputs.append(httpclient.InferInput("INPUT0", input_data_0.shape, "FP32"))
# inputs[0].set_data_from_numpy(input_data_0)

# outputs = [
#     httpclient.InferRequestedOutput("OUTPUT0"),
# ]

# response = triton_client.infer(model_name,
#                                inputs,
#                                request_id=str(1),
#                                outputs=outputs,
#                                timeout=1800000
#                                )

# # Process and print the output
# output_0 = response.as_numpy("OUTPUT0")
# print("Number of segments:", output_0)


##############################################################################
# Superpixel Segmentation with SLIC using Triton Inference Server and Python backend
# Author: Ahmad
# Date: 2023-07-26
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
from skimage.segmentation import slic
from cuda_slic.slic import slic as cuda_slic
from large_image.cache_util import cachesClear
import time
import openslide
import time


def _callback(capture, result, error):
    """Callback for async_infer to capture result or error of inference
    request.

    Parameters
    ----------
    capture : list
        An empty list of
    result : grpcclient.InferResult
        The result of inference if successful.
    error : tritonclientutils.InferenceServerException
        An exception if inference failed. Otherwise None.
    """

    if error:
        capture.append((error, time.time()))
    else:
        capture.append((result, time.time()))


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


no_triton_test = True
triton_test = False
model_name = "superpixel_slic_noclient"
tile_size = (4096, 4096)
time_elapsed_slic_notriton = 0
time_elapsed_cuda_slic_notriton = 0
time_elapsed_triton = 0
num_tiles = 0

slide_path = "/tf/notebook/TCGA-TM-A7CF-01Z-00-DX1.EB905EDB-5AC5-41C4-AB00-526DD3820524.svs"

slide = openslide.OpenSlide(slide_path)

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
print("total_tiles:",total_tiles)

tile_size = (4096, 4096)
x_start, y_start = 0, 0
x_end, y_end = x_start + tile_size[0], y_start + tile_size[1]
tile = slide.read_region((x_start, y_start), 0, (x_end - x_start, y_end - y_start))
tile = np.array(tile)[:, :, :3]  # Convert RGBA to RGB
# Convert the tile to float32 for SLIC
input_data_0 = np.array(tile, dtype=np.float32)
print(input_data_0.shape)


# Perform SLIC segmentation on the input image
n_segments = 125


if (no_triton_test):

    # Iterate through the first x tiles
    for y in range(total_tiles_height):
        for x in range(total_tiles_width):
            x_start, y_start = x * tile_size[0], y * tile_size[1]
            x_end, y_end = x_start + tile_size[0], y_start + tile_size[1]
            tile = slide.read_region((x_start, y_start), 0, (x_end - x_start, y_end - y_start))
            tile = np.array(tile)[:, :, :3]  # Convert RGBA to RGB
            input_data_0 = np.array(tile, dtype=np.float32)

            # Perform SLIC segmentation using scikit-image
            start_time = time.time()
            segments_slic = slic(input_data_0, n_segments=n_segments, compactness=10, sigma=1)
            end_time = time.time()
            time_elapsed_slic_notriton += (end_time - start_time)

            # Perform SLIC segmentation using custom CUDA SLIC implementation
            start_time = time.time()
            segments_slic_cuda = cuda_slic(input_data_0, n_segments=n_segments, max_iter=5)
            end_time = time.time()
            time_elapsed_cuda_slic_notriton += (end_time - start_time)

            num_tiles += 1
            print("num_tiles:", num_tiles)

            if num_tiles >= 20:
                break


    # Define the number of segments for SLIC

    # Calculate time elapsed

    throughput_slic_notriton = num_tiles / time_elapsed_slic_notriton
    throughput_cuda_slic_notriton = num_tiles / time_elapsed_cuda_slic_notriton
    print("Time Elapsed slic without triton: {:.2f} seconds".format(time_elapsed_slic_notriton))
    print("Throughput slic without triton: {:.2f} seconds".format(throughput_slic_notriton))

    print("Time Elapsed cuda slic without triton: {:.8f} seconds".format(time_elapsed_cuda_slic_notriton))
    print("Throughput cuda slic without triton: {:.8f} seconds".format(throughput_cuda_slic_notriton))


if (triton_test):


    # Create Triton HTTP client
    triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
    triton_client_grpc = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

    # Load model
    try:
        triton_client.unload_model(model_name)
        triton_client.load_model(model_name)                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        print(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"Failed to load model '{model_name}': {e}")

    # Create grpc InferenceRequest
    inputs = []
    inputs.append(grpcclient.InferInput("INPUT0", (4096,4096,3), "FP32"))
    # inputs[0].set_data_from_numpy(input_data_0)
    # outputs = [
    #     grpcclient.InferRequestedOutput("OUTPUT0"),
    # ]
    client_nogpu()
    cache_clear()
    gpu_mem_clear()
    num_tiles = 0
    start_time = time.time()
    for y in range(total_tiles_height):
        for x in range(total_tiles_width):
            x_start, y_start = x * tile_size[0], y * tile_size[1]
            x_end, y_end = x_start + tile_size[0], y_start + tile_size[1]
            tile = slide.read_region((x_start, y_start), 0, (x_end - x_start, y_end - y_start))
            tile = np.array(tile)[:, :, :3]  # Convert RGBA to RGB
            input_data_0 = np.array(tile, dtype=np.float32)

            # inputs = []
            inputs0 = []
            sample = {'result': []}
            inputs0.append(grpcclient.InferInput("INPUT0", (4096,4096,3), "FP32"))
            inputs0[0].set_data_from_numpy(input_data_0)
            outputs = [
                grpcclient.InferRequestedOutput("OUTPUT0"),
            ]

            # Measure the  time taken for inference with Skimage-SLIC or CUDA-SLIC
            
            response = triton_client_grpc.async_infer(model_name, inputs=inputs0, callback=functools.partial(_callback, sample["result"]), outputs=outputs)
            num_tiles += 1
            print("num_tiles:", num_tiles)
            # if num_tiles >= 2:
            #     break
    end_time = time.time()
    time_elapsed_triton += (end_time - start_time)

    
    print("Time Elapsed Skimage-SLIC with triton only: {:.8f} seconds".format(time_elapsed_triton))

    # Calculate throughput (tiles per second)
    # use total_tiles  for total number of tiles
    print("num_tiles:",num_tiles)
    throughput = num_tiles / time_elapsed_triton
    print("Throughput Skimage-SLIC without triton client: {:.8f} tiles/second".format(throughput))

    print("total_tiles:",total_tiles)
    throughput = total_tiles / time_elapsed_triton
    print("Throughput Skimage-SLIC without triton client: {:.8f} tiles/second".format(throughput))

