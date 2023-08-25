import argparse
import math
import triton_python_backend_utils as pb_utils
from skimage.transform import resize
import numpy as np
import json
from skimage.segmentation import slic
from cuda_slic.slic import slic as cuda_slic
import numpy
from histomicstk.cli import utils


def createSuperPixels_histomicstk(opts, input_np):
    print("spopts", opts)
    print(">> Reading input images")
    try:
        tile = {
            "tile": input_np,  # Your single tile data
            "tile_position": {
                "region_y": 0,  # Adjust this as needed
            },
        }
        averageSize = opts.superpixelSize**2
        overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0
        tileSize = opts.tileSize + overlap

    except Exception as inst:
        print(type(inst))  # the exception type
    strips = []

    img = tile["tile"]  # The single tile you have computed

    if overlap:
        mask = numpy.ones(img.shape[:2])
        try:
            for y, simg in strips:
                if (
                    y < ty0 + tile["height"]
                    and y + simg.height > ty0
                    and simg.width > tx0
                ):
                    suby = max(0, y - ty0)
                    subimg = simg.crop(
                        tx0,
                        max(0, ty0 - y),
                        min(tile["width"], simg.width),
                        min(tile["height"], simg.height - max(0, ty0 - y)),
                    )
                    # Our mask is true when a pixel has not been set
                    submask = (
                        numpy.ndarray(
                            buffer=subimg[3].write_to_memory(),
                            dtype=numpy.uint8,
                            shape=[subimg.height, subimg.width],
                        )
                        == 0
                    )
                    mask[suby : suby + submask.shape[0], : submask.shape[1]] *= submask
            n_pixels = numpy.count_nonzero(mask)
        except Exception as e:
            print("An error occurred:", e)
    print("n_pixels", n_pixels)

    n_segments = math.ceil(n_pixels / averageSize)
    cuda_segments_slic = cuda_slic(img, n_segments=n_segments, max_iter=5)

    return cuda_segments_slic


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "output")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
    def execute(self, requests):
        responses = []
        total_tile_count = len(requests)
        total_elapsed_time_cuda = 0
        total_num_pixels = 0
        verbos = False
        cuda_slic_test = True
        RUN = False
        spopts = argparse.Namespace(
            roi=[-1, -1, -1, -1],
            tileSize=4096,
            superpixelSize=100,
            magnification=5,
            overlap=True,
            boundaries=True,
            bounding="Internal",
            slic_zero=True,
            compactness=0.1,
            sigma=1,
            default_category_label="default",
            default_fillColor="rgba(0, 0, 0, 0)",
            default_strokeColor="rgba(0, 0, 0, 1)",
        )

        for request in requests:
            print("Python backlend execute")
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input")
            input_np = input_tensor.as_numpy()
            # Handle different input shapes
            if input_np.shape[0] == 1:
                # Handle [3, 4096, 4096, 3] input shape
                input_np = input_np[0, :, :, :]
                # print("Received input shape [2, 4096, 4096, 3]")
            else:
                raise ValueError("Invalid input shape")

            n_segments = 125

            print("input")

            cuda_segments_slic = createSuperPixels_histomicstk(spopts, input_np)


            out_tensor_cuda = pb_utils.Tensor(
                "output", cuda_segments_slic.astype(np.float32)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_cuda]
            )

            responses.append(inference_response)

            # Calculate tile shape and total throughput for CUDA SLIC

            if cuda_slic_test:
                if verbos:
                    total_throughput_cuda = total_num_pixels / total_elapsed_time_cuda
                    throughput_tiles_cuda = total_tile_count / total_elapsed_time_cuda
                    print("            ----------------------------------            ")
                    print(
                        "Final Total elapsed time for CUDA SLIC:",
                        total_elapsed_time_cuda,
                        "seconds",
                    )
                    print(
                        "Final Total throughput for CUDA SLIC (pixels/second):",
                        total_throughput_cuda,
                    )
                    print(
                        "Final Throughput for CUDA SLIC in terms of tiles/sec:",
                        throughput_tiles_cuda,
                        "tiles/second",
                    )

        return responses

    def finalize(self):
        print("Cleaning up...")
