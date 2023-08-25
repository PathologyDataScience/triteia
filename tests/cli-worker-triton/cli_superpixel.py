import numpy as np
import skimage.segmentation
import openslide
import argparse
import large_image
import histomicstk
import numpy
import scipy
import pyvips
import json
import os
from pathlib import Path
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from simple_triton.feature_extraction import histomics_stream_inference
from histomicstk.cli import utils
from mil.io.utils import study
import time
from simple_triton.model import TritonModel
from simple_triton.config import ConfigBuilder


def perform_slic_segmentation(image, n_segments, compactness, sigma):
    segments = skimage.segmentation.slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        enforce_connectivity=True,
    )
    return segments


def perform_seam_overlapping(segments, overlap_amount):
    overlapped_segments = segments.copy()
    for i in range(overlap_amount):
        overlapped_segments[:, i] = segments[:, 0]
    return overlapped_segments


def create_load_model(model_name, url):
    """The function feature_extractor can be used to create feature extraction
    models in the model repository. Note - this cell will take time as the model
    is downloaded, saved, and loaded into triton.

    Parameters in this stage include the inference server (address), the model (model name,
    maximum batch size).

    Args:
        client (tritonclient.grpc.InferenceServerClient):
        args_dict (dict): The inputs to the model from argparse.
        maxBatchSize (int): max batch size to for config
    """
    # slide paramters

    maxBatchSize = 1  # set max batch size
    count = 1  # set gpu count
    kind = "gpu"  # set gpu kind
    gpus = 8  # set number of gpus
    model = TritonModel(model_name, url)

    config = model.get_config()
    config_builder = ConfigBuilder(model_name=model_name, config=config, url=url)
    # Add/remove an automatic mixed-precision accelerator to the config.
    config_builder.add_mixed_precision()

    if kind == "gpu":
        start, end, intval = 0, gpus, 1
        gpus = list(range(start, end, intval))
        config_builder.remove_instance_groups()
        config_builder.add_instance_group(count, kind, gpus)

    config_builder.response_cache(False)
    model.unload(model_name)
    model.load(config=config_builder.config)
    print(model.get_config())
    assert model.is_loaded()


def createSuperPixels(opts, response, tile_info, meta):
    RUN = False

    def iterate_tiles(response, tile_info):
        tile_data_list = response[0]  # Extract the list of tile arrays from response
        for i, tile_data in enumerate(tile_data_list):
            meta_data = {
                "tile_height": tile_info["tile_height"][i],
                "magnification": tile_info["target_magnification"][i],
                "tile_width": tile_info["tile_width"][i],
                "overlap_height": tile_info["overlap_height"][i],
                "overlap_width": tile_info["overlap_width"][i],
                "filename": tile_info["filename"][i],
                "slide_name": tile_info["slide_name"][i],
                "chunk_top": tile_info["chunk_top"][i],
                "chunk_left": tile_info["chunk_left"][i],
                "chunk_bottom": tile_info["chunk_bottom"][i],
                "chunk_right": tile_info["chunk_right"][i],
                "tile_top": tile_info["tile_top"][i],
                "tile_left": tile_info["tile_left"][i],
            }

            tile_position = {
                "top": meta_data["chunk_top"],
                "left": meta_data["chunk_left"],
                "bottom": meta_data["chunk_bottom"],
                "right": meta_data["chunk_right"],
            }

            yield {
                "tile_data": tile_data,
                "meta_data": meta_data,
                "tile_position": tile_position,
            }

    averageSize = opts.superpixelSize**2
    overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0
    tileSize = opts.tileSize + overlap
    strips = []
    found = 0
    bboxes = []
    bboxesUser = []
    tiparams = {}
    scale = 1
    tiparams = utils.get_region_dict(
        opts.roi, None
    )  # You might need to pass other arguments to this function
    if opts.magnification:
        tiparams["scale"] = {"magnification": opts.magnification}

    iter = 0
    for tile_info_dict in iterate_tiles(response, tile_info):
        tile_data = tile_info_dict["tile_data"]
        meta_data = tile_info_dict["meta_data"]
        tile_position = tile_info_dict["tile_position"]
        tile_position = tile_info_dict["tile_position"]

        # print("Tile Data:", tile_data)
        print("Meta Data:", meta_data)
        print("Tile Position:", tile_position)
        print("-----")

        print(">> Generating superpixels")
        if opts.slic_zero:
            print(">> Using SLIC Zero for segmentation")

        if meta["magnification"] and meta_data["magnification"]:
            scale = meta["magnification"] / meta_data["magnification"]

        x0 = tiparams.get("region", {}).get("left", 0)
        y0 = tiparams.get("region", {}).get("top", 0)

        tx0 = int((meta_data["tile_left"] - x0) / scale)
        ty0 = int((meta_data["tile_top"] - y0) / scale)
        # img = meta_data['tile']
        n_pixels = meta_data["tile_width"] * meta_data["tile_height"]
        mask = None
        # Extract other meta_data values as needed

        # Overlap and processing code
        if overlap:
            mask = np.ones(tile_data.shape[:2])
            # creating and applying a mask on the superpixel segments to ensure that only certain parts of
            # the image contribute to the superpixel generation process.
            if strips is not None:
                try:
                    for y, simg in strips:
                        if (
                            y < ty0 + meta_data["tile_height"]
                            and y + simg.height > ty0
                            and simg.width > tx0
                        ):
                            suby = max(0, y - ty0)
                            subimg = simg.crop(
                                tx0,
                                max(0, ty0 - y),
                                min(meta_data["tile_width"], simg.width),
                                min(
                                    meta_data["tile_height"],
                                    simg.height - max(0, ty0 - y),
                                ),
                            )
                            submask = (
                                numpy.ndarray(
                                    buffer=subimg[3].write_to_memory(),
                                    dtype=numpy.uint8,
                                    shape=[subimg.height, subimg.width],
                                )
                                == 0
                            )
                            mask[
                                suby : suby + submask.shape[0], : submask.shape[1]
                            ] *= submask
                    n_pixels = np.count_nonzero(mask)
                except Exception as e:
                    print("An error occurred:", e)
            # Rest of the overlap processing code
            maxValue = numpy.max(tile_data) + 1
            if overlap:
                # Keep any segment that is at all in the non-overlap region
                core = tile_data[
                    : meta_data["tile_height"] - meta_data["overlap_width"],
                    : meta_data["tile_width"] - meta_data["overlap_height"],
                ]
                coremask = mask[
                    : meta_data["tile_height"] - meta_data["overlap_width"],
                    : meta_data["tile_width"] - meta_data["overlap_height"],
                ]
                core[numpy.where(coremask != 1)] = -1
                usedIndices = numpy.unique(core)
                usedIndices = numpy.delete(usedIndices, numpy.where(usedIndices < 0))
                usedLut = [-1] * int(maxValue)
                for idx, used in enumerate(usedIndices):
                    if used >= 0:
                        usedLut[int(used)] = idx
                usedLut = numpy.array(usedLut, dtype=int)
                print("reduced from %d to %d" % (maxValue, len(usedIndices)))
                maxValue = len(usedIndices)
                # segments = usedLut[tile_data]
                segments = usedLut[tile_data.astype(int)]
                mask *= segments != -1
            if str(opts.bounding).lower() not in {"", "none"}:
                regions = skimage.measure.regionprops(1 + segments)
                for pidx, props in enumerate(regions):
                    by0, bx0, by1, bx1 = props.bbox
                    bboxes.append(
                        (
                            ((bx0 + bx1) / 2 + tx0) * scale + x0,
                            ((by0 + by1) / 2 + ty0) * scale + y0,
                            (bx1 - bx0) * scale,
                            (by1 - by0) * scale,
                        )
                    )
                    bboxesUser.extend(
                        [
                            (bx0 + tx0) * scale + x0,
                            (by0 + ty0) * scale + y0,
                            (bx1 + tx0) * scale + x0,
                            (by1 + ty0) * scale + y0,
                        ]
                    )
            if opts.boundaries:
                segments *= 2
                maxValue *= 2
                edges = (scipy.ndimage.sobel(segments, axis=0) != 0) | (
                    scipy.ndimage.sobel(segments, axis=1) != 0
                )
                edges[0, :] = True
                edges[-1, :] = True
                edges[:, 0] = True
                edges[:, -1] = True
                segments += edges
            segments += found
            found += int(maxValue)
            if mask is None:
                data = numpy.dstack(
                    (
                        (segments % 256).astype(int),
                        (segments / 256).astype(int) % 256,
                        (segments / 65536).astype(int) % 256,
                    )
                ).astype("B")
            else:
                data = numpy.dstack(
                    (
                        (segments % 256).astype(int),
                        (segments / 256).astype(int) % 256,
                        (segments / 65536).astype(int) % 256,
                        mask * 255,
                    )
                ).astype("B")
            # For overlay, suppose we make any value whose centroid is in the
            # overlap region transparent.  Then, use vips to overlap the
            # images rather than inserting them.
            vimg = pyvips.Image.new_from_memory(
                numpy.ascontiguousarray(data).data,
                data.shape[1],
                data.shape[0],
                data.shape[2],
                large_image.constants.dtypeToGValue[data.dtype.char],
            )
            vimg = vimg.copy(interpretation=pyvips.Interpretation.RGB)
            vimgTemp = pyvips.Image.new_temp_file("%s.v")
            vimg.write(vimgTemp)
            vimg = vimgTemp
            x = tx0
            ty = tile_position["right"]
            while len(strips) <= ty:
                strips.append(None)
            if strips[ty] is None:
                strip = pyvips.Image.black(
                    tiparams.get("region", {}).get("width", meta["sizeX"]),
                    vimg.height,
                    bands=vimg.bands,
                )
                strip = strip.copy(interpretation=pyvips.Interpretation.RGB)
                strips[ty] = [ty0, strip]
            strips[ty][1] = strips[ty][1].composite(
                [vimg], pyvips.BlendMode.OVER, x=int(x), y=0
            )
            if hasattr(opts, "callback"):
                opts.callback(
                    "tiles",
                    meta_data["tile_position"]["position"] + 1,
                    meta_data["iterator_range"]["position"],
                )
        iter += 1
        # if iter > 10:
        #     break

    if hasattr(opts, "callback"):
        opts.callback("file", 0, 2 if opts.outputAnnotationFile else 1)
    print(">> Found %d superpixels" % found)
    if found > 256**3:
        print("Too many superpixels")

    if strips[0] is not None:
        bands = strips[0][1].bands
    else:
        bands = 3

    img = pyvips.Image.black(
        tiparams.get("region", {}).get("width", meta["sizeX"]) / scale,
        tiparams.get("region", {}).get("height", meta["sizeY"]) / scale,
        # bands=strips[0][1].bands)
        bands=bands,
    )
    img = img.copy(interpretation=pyvips.Interpretation.RGB)
    if RUN:
        for stripidx in range(len(strips)):
            img = img.composite(
                [strips[stripidx][1]],
                pyvips.BlendMode.OVER,
                x=0,
                y=int(strips[stripidx][0]),
            )
    # Discard alpha band, if any.
    img = img[:3]
    # Add program run parameters to the image description and list the
    # superpixel count
    img.set_type(
        pyvips.GValue.gstr_type,
        "image-description",
        json.dumps(
            dict(
                {k: v for k, v in vars(opts).items() if k != "callback"},
                indexCount=found,
            )
        ),
    )
    img.write_to_file(
        opts.outputImageFile,
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        region_shrink=pyvips.RegionShrink.NEAREST,
        # We'd prefer max, but to do so we need to compute max of the
        # superpixel, not the faux-color it is mapped to.
        # region_shrink=pyvips.RegionShrink.MAX,
        bigtiff=True,
        compression="lzw",
        predictor="horizontal",
    )

    if hasattr(opts, "callback"):
        opts.callback("file", 1, 2 if opts.outputAnnotationFile else 1)
    # Annotation code
    if opts.outputAnnotationFile:
        categories = [
            {
                "label": opts.default_category_label,
                "fillColor": opts.default_fillColor,
                "strokeColor": opts.default_strokeColor,
            },
        ]
        annotation_name = os.path.splitext(os.path.basename(opts.outputAnnotationFile))[
            0
        ]
        region_dict = utils.get_region_dict(opts.roi, None)
        annotation = {
            "name": annotation_name,
            "elements": [
                {
                    "type": "pixelmap",
                    "girderId": "outputImageFile",
                    "transform": {
                        "xoffset": region_dict.get("region", {}).get("left", 0) / scale,
                        "yoffset": region_dict.get("region", {}).get("top", 0) / scale,
                        "matrix": [[scale, 0], [0, scale]],
                    },
                    "values": [0] * (found // (2 if opts.boundaries else 1)),
                    "categories": categories,
                    "boundaries": opts.boundaries,
                }
            ],
            "attributes": {
                "params": vars(opts),
                "cli": Path(__file__).stem,
                "version": histomicstk.__version__,
            },
        }
        # Rest of the annotation code

        with open(opts.outputAnnotationFile, "w") as annotation_file:
            json.dump(
                annotation, annotation_file, separators=(",", ":"), sort_keys=False
            )
        if hasattr(opts, "callback"):
            opts.callback("file", 2, 2)


def create_hs_study(wsi_path, tile, chunk, target_magnification=20, source="exact"):
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


def main():
    # Load large image using openslide from a file path
    image_path = (
        "/tf/notebook/TCGA-TM-A7CF-01Z-00-DX1.EB905EDB-5AC5-41C4-AB00-526DD3820524.svs"
    )
    slide_path = [
        "/tf/notebook/TCGA-TM-A7CF-01Z-00-DX1.EB905EDB-5AC5-41C4-AB00-526DD3820524.svs"
    ]
    tempdir = "/tf/notebook/"
    outImagePath1 = "outputImageFile.tiff"
    outImagePath = os.path.join(tempdir, "superpixel.tiff")
    annotationName = "Superpixel"
    outAnnotationPath = os.path.join(tempdir, "%s.anot" % annotationName)
    model_name = "superpixel_overlap"
    tiles_size = (4096, 4096)
    tile_size = 4096
    RUN = False

    ts = large_image.open(image_path)
    meta = ts.getMetadata()
    print("meta", meta)

    output_image_dir = "/tf/notebook/temp/python_backend/examples/cuda_slic"
    os.makedirs(output_image_dir, exist_ok=True)

    slide = openslide.OpenSlide(image_path)

    (width, height) = slide.dimensions
    factors = slide.level_downsamples

    total_tiles_width = width // tiles_size[0]
    total_tiles_height = height // tiles_size[1]
    total_tiles = total_tiles_width * total_tiles_height

    # Get tile shape and batch size
    tile_shape = (4096, 4096, 3)
    batch_size = 1

    spopts = argparse.Namespace(
        inputImageFile=image_path,
        outputImageFile=outImagePath,
        outputAnnotationFile=outAnnotationPath,
        roi=[-1, -1, -1, -1],
        tileSize=4096,
        superpixelSize=100,
        magnification=20,
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
    # Create Triton HTTP client
    triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    triton_client_grpc = grpcclient.InferenceServerClient(
        url="localhost:8001", verbose=True
    )

    try:
        triton_client.unload_model(model_name)
        triton_client.load_model(model_name)
        # create_load_model(model_name,"localhost:8001")
        print(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"Failed to load model '{model_name}': {e}")

    # Create a histomic-stream study
    print("Create a histomic-stream study")
    hs_study = create_hs_study(slide_path, tile_size, tile_size, 20, "exact")

    outputs = [grpcclient.InferRequestedOutput("output")]
    output = [httpclient.InferRequestedOutput("output")]

    # Perform custom histomics stream inference
    print("Perform custom histomics stream inference")

    num_iterations = 1
    average_throughput_tiles_cuda = 0

    # createSuperPixels_histomicstk(spopts,segment,tiles)

    for _ in range(num_iterations):
        start_time = time.time()
        response, tile_info, times, failed = histomics_stream_inference(
            hs_study, model_name, url="localhost:8001", batch=1, workers=32, limit=10
        )
        # response = triton_client_grpc.infer(model_name, inputs=inputs0, outputs=outputs)
        print("response:{} tile_info", response)
        elapsed_time = time.time() - start_time
        throughput_tiles_cuda = total_tiles / elapsed_time
        average_throughput_tiles_cuda += throughput_tiles_cuda
    average_throughput_tiles_cuda /= num_iterations

    print("            ----------------------------------            ")
    print(
        "Average Throughput for CUDA-SLICin terms of tiles/sec:",
        average_throughput_tiles_cuda,
        "tiles/second",
    )
    print(
        "Average elapsed_time for CUDA-SLICin terms of seconds:",
        elapsed_time,
        "seconds",
    )

    # Create a dictionary to map tile IDs to tuples of tile data and metadata
    ts = large_image.open(image_path)
    meta = ts.getMetadata()
    print("meta", meta)
    createSuperPixels(spopts, response, tile_info, meta)

    print("FINISHED Inference")

    slide.close()


if __name__ == "__main__":
    main()
