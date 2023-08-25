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

import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from mil.io.utils import study
from skimage.io import imread
from simple_triton.feature_extraction import histomics_stream_inference
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize
from skimage.segmentation import slic
import time
import openslide
import time

from simple_triton.feature_extraction import histomics_stream_inference


def output_tensor():
    try:
        # Make the inference request
        response = triton_client_grpc.infer(model_name, inputs, request_id=str(1), outputs=outputs, timeout=1800000)

        # Process the response
        if response is not None:
            # Assuming the response has a single output tensor named "OUTPUT0"
            output_tensor = response.as_numpy("OUTPUT0")
            # Do something with the output tensor, e.g., post-processing or saving the results

            # Return the output tensor or results to the caller
            return output_tensor
        else:
            # Handle the case when the response is None or empty
            print("Empty response received from the server.")
            return None

    except Exception as e:
        # Handle exceptions that might occur during inference
        print(f"Error during inference: {e}")
        return None
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
    # slide parameters
    
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
    # ... (any other required code for setup or pre-processing)

    # Call the original histomics_stream_inference with the provided inputs and outputs
    return histomics_stream_inference(hs_study, model_name, url=url, batch=batch, workers=workers, limit=limit)
   
model_name = "superpixel_slic"

slide_path = ["/tf/notebook/TCGA-TM-A7CF-01Z-00-DX1.EB905EDB-5AC5-41C4-AB00-526DD3820524.svs"]
# desired_shape = (4096, 4096, 3)

# slide = openslide.OpenSlide(slide_path)

# magnification = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
# (width, height) = slide.dimensions
# factors = slide.level_downsamples

# Display metadata
# print("slide scanned at {} objective magnification".format(magnification))
# print("scanned image is {0} x {1} pixels".format(width, height))
# print("contains {0} levels with downsample factors: {1}".format(len(factors), factors))

tile_size = (4096, 4096)
tile = 4096
num_tiles = 10  # Define the number of tiles to process

# Define the number of segments for SLIC
n_segments = 125

# Create Triton HTTP client
triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
triton_client_grpc = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

# use multi-tiles reading
# multi workers to read tiles and do inference runner
# Use parameters used by Superpixel classificiation software

# Load model
try:
    # triton_client.unload_model(model_name)
    triton_client.load_model(model_name)
    print(f"Model '{model_name}' loaded successfully.")
except Exception as e:
    print(f"Failed to load model '{model_name}': {e}")

# Convert the tile to float32 for SLIC
# input_data_0 = np.array(tile, dtype=np.float32)


hs_study = create_hs_study(slide_path, tile, tile,  20, "exact")

# np_array = np.array(hs_study, dtype=np.float32)

# # Assuming values_list contains the tile data from hs_study
# values_list = [value for value in hs_study.values() if isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value)]

# # Convert the filtered list to a NumPy array with dtype=float32
# # Convert the filtered list to a NumPy array with dtype=float32
# np_array = np.array(values_list, dtype=np.float32)

# # Desired shape of the final array
# desired_shape = (4096, 4096, 3)

# print("Size of np_array:", np_array.size)

# # Check if np_array is empty
# if np_array.size == 0:
#     print("No tile data found in hs_study. Skipping resizing.")
#     np_array_resized = None
# else:
#     # Calculate the number of times to repeat the array to fill the desired shape
#     repeat_rows = desired_shape[0] // np_array.shape[0]
#     repeat_columns = desired_shape[1] // np_array.shape[1]

#     # Replicate the array to match the desired shape
#     np_array_resized = np.tile(np_array, (repeat_rows, repeat_columns, 1))

#     # If the number of tiles is not divisible by the desired number of rows/columns, we need to crop the array
#     np_array_resized = np_array_resized[:desired_shape[0], :desired_shape[1], :]

# # Rest of the code follows...



           
# Create the request inputs and outputs
# inputs = []
# empty_array = np.empty((4096, 4096, 3), dtype=np.float32)
# inputs.append(grpcclient.InferInput("INPUT0", (4096, 4096, 3), "FP32"))
# inputs[0].set_data_from_numpy(empty_array)

outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]

(
    response,
    tile_info,
    times,
    failed,
) = custom_histomics_stream_inference(
    hs_study,
    model_name,
    url="localhost:8001",
    batch=64,
    workers=32,
    limit=10,

)
print("FINISHEED Inference")

# response = output_tensor()
# Perform SLIC segmentation on N number of tiles
# for i in range(num_tiles):
#     x_start, y_start = i * tile_size[0], 0
#     x_end, y_end = x_start + tile_size[0], y_start + tile_size[1]
#     tile = slide.read_region((x_start, y_start), 0, (x_end - x_start, y_end - y_start))
#     tile = np.array(tile)[:, :, :3]  # Convert RGBA to RGB

#     # Perform SLIC segmentation on the input image
#     # segments_slic = slic(input_data_0, n_segments=n_segments, compactness=10, sigma=1)

#     # Create grpc InferenceRequest
    
#     inputs = []
#     inputs.append(grpcclient.InferInput("INPUT0", input_data_0.shape, "FP32"))
#     inputs[0].set_data_from_numpy(input_data_0)
#     outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]

    # Measure the time taken for inference
# start_time = time.time()try:
    # Make the inference request

# end_time = time.time()

#     # Calculate time elapsed
#     time_elapsed = end_time - start_time
#     print("Time Elapsed for processing tile {}: {:.2f} seconds".format(i, time_elapsed))

#     # Calculate throughput (segments per second)
#     throughput = n_segments / time_elapsed
#     print("Throughput for processing tile {}: {:.2f} segments/second".format(i, throughput))



