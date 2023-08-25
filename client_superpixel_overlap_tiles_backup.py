import numpy as np
import skimage.segmentation
import matplotlib.pyplot as plt
import openslide
import argparse
import math
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
from cuda_slic.slic import slic as cuda_slic
from simple_triton.model import TritonModel
from simple_triton.config import ConfigBuilder
import pickle

def perform_slic_segmentation(image, n_segments, compactness, sigma):
    segments = skimage.segmentation.slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        enforce_connectivity=True
    )
    return segments

def perform_seam_overlapping(segments, overlap_amount):
    overlapped_segments = segments.copy()
    for i in range(overlap_amount):
        overlapped_segments[:, i] = segments[:, 0]
    return overlapped_segments


def create_load_model(model_name,url):
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
        kind = "gpu" # set gpu kind
        gpus = 8  # set number of gpus
        model = TritonModel(model_name, url)
        # model.load({"maxBatchSize": maxBatchSize})
        # assert model.is_loaded()
        config = model.get_config()
        config_builder = ConfigBuilder(model_name=model_name, config=config, url=url)
        # Add/remove an automatic mixed-precision accelerator to the config.
        config_builder.add_mixed_precision()
        # Add an TensorRT accelerator to the config.
        #  Add an instance group defining the model hardware resources and instances.
        if kind == "gpu":
            start, end, intval = 0, gpus, 1
            gpus = list(range(start, end, intval))
            config_builder.remove_instance_groups()
            config_builder.add_instance_group(count, kind, gpus)
        # load tensorflow model with larger batch size
        # config_builder.max_batch_size(maxBatchSize)
        config_builder.response_cache(False)
        model.unload(model_name)
        # triton_client_grpc.unload_model(model_name)
        model.load(config=config_builder.config)
        print(model.get_config())
        assert model.is_loaded()


# def createSuperPixels(opts):
#     found = 0
    
#     # Calculate the average number of pixels in a superpixel
#     averageSize = opts.superpixelSize ** 2

#     # Calculate the overlap between tiles
#     overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0

#     # Calculate the effective tile size considering the overlap
#     tileSize = opts.tileSize + overlap

#     print('>> Reading input image')
#     print(opts.inputImageFile)

#     ts = large_image.open(opts.inputImageFile)
#     meta = ts.getMetadata()
#     found = 0
#     strips = []
#     bboxes = []
#     bboxesUser = []
#     tiparams = {}
#     # Handle the absence of opts.roi attribute
#     if hasattr(opts, 'roi'):
#         tiparams = utils.get_region_dict(opts.roi, None, ts)
#     else:
#         tiparams = None  # Set a default value or adjust based on your requirements

#     # Inside the createSuperPixels function
#     if tiparams is not None:
#         for tile in ts.tileIterator(
#             format=large_image.constants.TILE_FORMAT_NUMPY,
#             tile_size=dict(width=tileSize, height=tileSize),
#             tile_overlap=dict(x=overlap, y=overlap),
#             **tiparams,
#         ):
#             print('>> Generating superpixels')
#         if opts.slic_zero:
#             print('>> Using SLIC Zero for segmentation')
#             # Iterate through the tiles of the image
#         for tile in ts.tileIterator(
#             format=large_image.constants.TILE_FORMAT_NUMPY,
#             tile_size=dict(width=tileSize, height=tileSize),
#             tile_overlap=dict(x=overlap, y=overlap),
#             **tiparams,
#         ):

#     # Check if 'callback' attribute exists in options
#             if hasattr(opts, 'callback'):
#             # Call the 'callback' function to signal file processing progress
#                 opts.callback('file', 0, 1)
#             # Rest of the loop code
#     else:
#         for tile in ts.tileIterator(
#             format=large_image.constants.TILE_FORMAT_NUMPY,
#             tile_size=dict(width=tileSize, height=tileSize),
#             tile_overlap=dict(x=overlap, y=overlap),
#         ):
#             # Rest of the loop code



#             # Print the number of superpixels found
#             print('>> Found %d superpixels' % found)

#             # Check if the number of found superpixels exceeds a threshold
#             if found > 256 ** 3:
#                 print('Too many superpixels')


def createSuperPixels(opts, response, tile_info, meta):  
    
    RUN = False
    def iterate_tiles(response, tile_info):
     tile_data_list = response[0]  # Extract the list of tile arrays from response
     for i, tile_data in enumerate(tile_data_list):
        
        
        meta_data = {
            'tile_height': tile_info['tile_height'][i],
            'magnification': tile_info['target_magnification'][i],
            'tile_width': tile_info['tile_width'][i],
            'overlap_height': tile_info['overlap_height'][i],
            'overlap_width': tile_info['overlap_width'][i],
            'filename': tile_info['filename'][i],
            'slide_name': tile_info['slide_name'][i],
             'chunk_top': tile_info['chunk_top'][i],
            'chunk_left': tile_info['chunk_left'][i],
            'chunk_bottom': tile_info['chunk_bottom'][i],
            'chunk_right': tile_info['chunk_right'][i],
            'tile_top': tile_info['tile_top'][i],
            'tile_left': tile_info['tile_left'][i]
        }
        
        tile_position = {
            'top': meta_data['chunk_top'],
            'left': meta_data['chunk_left'],
            'bottom': meta_data['chunk_bottom'],
            'right': meta_data['chunk_right']
        }
        
        yield {
            'tile_data': tile_data,
            'meta_data': meta_data,
            'tile_position': tile_position
        }
        
        
    averageSize = opts.superpixelSize ** 2
    overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0
    tileSize = opts.tileSize + overlap
    strips = []    
    found = 0
    bboxes = []
    bboxesUser = []
    tiparams = {} 
    scale = 1
    tiparams = utils.get_region_dict(opts.roi, None)  # You might need to pass other arguments to this function
    if opts.magnification:
     tiparams['scale'] = {'magnification': opts.magnification}  
     
     
    iter = 0   
    for tile_info_dict in iterate_tiles(response, tile_info):
        tile_data = tile_info_dict['tile_data']
        meta_data = tile_info_dict['meta_data']
        tile_position = tile_info_dict['tile_position']
        tile_position = tile_info_dict['tile_position']
        
        # print("Tile Data:", tile_data)
        print("Meta Data:", meta_data)
        print("Tile Position:", tile_position)
        print("-----")
       
        print('>> Generating superpixels')
        if opts.slic_zero:
            print('>> Using SLIC Zero for segmentation')

        if meta['magnification'] and meta_data['magnification']:
            scale = meta['magnification'] / meta_data['magnification']
        
        x0 = tiparams.get('region', {}).get('left', 0)
        y0 = tiparams.get('region', {}).get('top', 0)

        
        tx0 = int((meta_data['tile_left'] - x0) / scale)
        ty0 = int((meta_data['tile_top']  - y0) / scale)
        # img = meta_data['tile']
        n_pixels = meta_data['tile_width'] * meta_data['tile_height']
        mask = None
        # Extract other meta_data values as needed

        # Rest of the code for superpixel generation, overlap, etc.
        # ...

        # Overlap and processing code
        if overlap:
            mask = np.ones(tile_data.shape[:2])
            #creating and applying a mask on the superpixel segments to ensure that only certain parts of 
            # the image contribute to the superpixel generation process.
            if strips is not None:
                try:
                    for y, simg in strips:
                        if (y < ty0 + meta_data['tile_height'] and y + simg.height > ty0 and simg.width > tx0):
                            suby = max(0, y - ty0)
                            subimg = simg.crop(
                                tx0,
                                max(0, ty0 - y),
                                min(meta_data['tile_width'], simg.width),
                                min(meta_data['tile_height'], simg.height - max(0, ty0 - y)))
                            # subimg = simg[
                            #     tile_data['tile_position']['left']: min(tile_width, simg.shape[1]),
                            #     max(0, tile_height - y): min(tile_data['tile_height'], simg.shape[0] - max(0, tile_height - y))]
                            # Our mask is true when a pixel has not been set
                            submask = numpy.ndarray(
                                buffer=subimg[3].write_to_memory(),
                                dtype=numpy.uint8,
                                shape=[subimg.height, subimg.width]) == 0
                            mask[suby:suby + submask.shape[0], :submask.shape[1]] *= submask
                            # submask = subimg[:, :, 3] == 0
                            # mask[suby:suby + submask.shape[0], :submask.shape[1]] *= submask
                # Update n_pixels using mask
                    n_pixels = np.count_nonzero(mask)
                except Exception as e:
                    print("An error occurred:", e)
            # Rest of the overlap processing code
            maxValue = numpy.max(tile_data) + 1
            if overlap:
                # Keep any segment that is at all in the non-overlap region
                core = tile_data[
                    :meta_data['tile_height'] - meta_data['overlap_width'],
                    :meta_data['tile_width'] - meta_data['overlap_height']]
                coremask = mask[
                    :meta_data['tile_height'] - meta_data['overlap_width'],
                    :meta_data['tile_width'] - meta_data['overlap_height']]
                core[numpy.where(coremask != 1)] = -1
                usedIndices = numpy.unique(core)
                usedIndices = numpy.delete(usedIndices, numpy.where(usedIndices < 0))
                usedLut = [-1] * int(maxValue)
                for idx, used in enumerate(usedIndices):
                    if used >= 0:
                        usedLut[int(used)] = idx
                usedLut = numpy.array(usedLut, dtype=int)
                print('reduced from %d to %d' % (maxValue, len(usedIndices)))
                maxValue = len(usedIndices)
                # segments = usedLut[tile_data]
                segments = usedLut[tile_data.astype(int)]
                mask *= (segments != -1)
            if str(opts.bounding).lower() not in {'', 'none'}:
                regions = skimage.measure.regionprops(1 + segments)
                for pidx, props in enumerate(regions):
                    by0, bx0, by1, bx1 = props.bbox
                    bboxes.append((
                        ((bx0 + bx1) / 2 + tx0) * scale + x0,
                        ((by0 + by1) / 2 + ty0) * scale + y0,
                        (bx1 - bx0) * scale,
                        (by1 - by0) * scale))
                    bboxesUser.extend([
                        (bx0 + tx0) * scale + x0,
                        (by0 + ty0) * scale + y0,
                        (bx1 + tx0) * scale + x0,
                        (by1 + ty0) * scale + y0,
                    ])
            if opts.boundaries:
                segments *= 2
                maxValue *= 2
                edges = (scipy.ndimage.sobel(segments, axis=0) != 0) | (
                    scipy.ndimage.sobel(segments, axis=1) != 0)
                edges[0, :] = True
                edges[-1, :] = True
                edges[:, 0] = True
                edges[:, -1] = True
                segments += edges
            segments += found
            found += int(maxValue)
            if mask is None:
                data = numpy.dstack((
                    (segments % 256).astype(int),
                    (segments / 256).astype(int) % 256,
                    (segments / 65536).astype(int) % 256)).astype('B')
            else:
                data = numpy.dstack((
                    (segments % 256).astype(int),
                    (segments / 256).astype(int) % 256,
                    (segments / 65536).astype(int) % 256,
                    mask * 255)).astype('B')
            # For overlay, suppose we make any value whose centroid is in the
            # overlap region transparent.  Then, use vips to overlap the
            # images rather than inserting them.
            vimg = pyvips.Image.new_from_memory(
                numpy.ascontiguousarray(data).data,
                data.shape[1], data.shape[0], data.shape[2],
                large_image.constants.dtypeToGValue[data.dtype.char])
            vimg = vimg.copy(interpretation=pyvips.Interpretation.RGB)
            vimgTemp = pyvips.Image.new_temp_file('%s.v')
            vimg.write(vimgTemp)
            vimg = vimgTemp
            x = tx0
            ty = tile_position['right']
            while len(strips) <= ty:
                strips.append(None)
            if strips[ty] is None:
                strip = pyvips.Image.black(
                    tiparams.get('region', {}).get('width', meta['sizeX']),
                    vimg.height, bands=vimg.bands)
                strip = strip.copy(interpretation=pyvips.Interpretation.RGB)
                strips[ty] = [ty0, strip]
            strips[ty][1] = strips[ty][1].composite([vimg], pyvips.BlendMode.OVER, x=int(x), y=0)
            if hasattr(opts, 'callback'):
                opts.callback('tiles', meta_data['tile_position']['position'] + 1,
                            meta_data['iterator_range']['position'])



        # ...

    if hasattr(opts, 'callback'):
        opts.callback('file', 0, 2 if opts.outputAnnotationFile else 1)
    print('>> Found %d superpixels' % found)
    if found > 256 ** 3:
        print('Too many superpixels')

    
    if strips[0] is not None:
        bands = strips[0][1].bands
    else:
        bands = 3
    
    
    img = pyvips.Image.black(
        tiparams.get('region', {}).get('width', meta['sizeX']) / scale,
        tiparams.get('region', {}).get('height', meta['sizeY']) / scale,
        # bands=strips[0][1].bands)
        bands=bands)
    img = img.copy(interpretation=pyvips.Interpretation.RGB)
    if(RUN):
        for stripidx in range(len(strips)):
            img = img.composite(
                [strips[stripidx][1]], pyvips.BlendMode.OVER, x=0, y=int(strips[stripidx][0]))
    # Discard alpha band, if any.
    img = img[:3]
    # Add program run parameters to the image description and list the
    # superpixel count
    img.set_type(
        pyvips.GValue.gstr_type, 'image-description',
        json.dumps(dict(
            {k: v for k, v in vars(opts).items() if k != 'callback'}, indexCount=found)))
    img.write_to_file(
        opts.outputImageFile, tile=True, tile_width=256, tile_height=256, pyramid=True,
        region_shrink=pyvips.RegionShrink.NEAREST,
        # We'd prefer max, but to do so we need to compute max of the
        # superpixel, not the faux-color it is mapped to.
        # region_shrink=pyvips.RegionShrink.MAX,
        bigtiff=True, compression='lzw', predictor='horizontal')

    if hasattr(opts, 'callback'):
        opts.callback('file', 1, 2 if opts.outputAnnotationFile else 1)
    # Annotation code
    if opts.outputAnnotationFile:
        categories = [
            {
                'label': opts.default_category_label,
                'fillColor': opts.default_fillColor,
                'strokeColor': opts.default_strokeColor,
            },
        ]
        annotation_name = os.path.splitext(os.path.basename(opts.outputAnnotationFile))[0]
        region_dict = utils.get_region_dict(opts.roi, None)
        annotation = {
            'name': annotation_name,
            'elements': [{
                'type': 'pixelmap',
                'girderId': 'outputImageFile',
                'transform': {
                    'xoffset': region_dict.get('region', {}).get('left', 0) / scale,
                    'yoffset': region_dict.get('region', {}).get('top', 0) / scale,
                    'matrix': [[scale, 0], [0, scale]],
                },
                'values': [0] * (found // (2 if opts.boundaries else 1)),
                'categories': categories,
                'boundaries': opts.boundaries,
            }],
            'attributes': {
                'params': vars(opts),
                'cli': Path(__file__).stem,
                'version': histomicstk.__version__,
            },
        }
        # Rest of the annotation code

        with open(opts.outputAnnotationFile, 'w') as annotation_file:
            json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)
        if hasattr(opts, 'callback'):
            opts.callback('file', 2, 2)


# def createSuperPixels_histomicstk(opts, input_np, meta):
    
#         # meta = {'levels': 11, 'sizeX': 175702, 'sizeY': 36385, 'tileWidth': 4096, 'tileHeight': 4096, 'magnification': 20.0, 'mm_x': 0.0002505, 'mm_y': 0.0002505, 'dtype': 'uint8', 'bandCount': 4}
#         # tile = {
#         #     'tile': input_np,  # Your single tile data
#         #     'width': meta['tileWidth'],  # Width of the tile
#         #     'height': meta['tileHeight'],  # Height of the tile
#         #     'magnification': meta['magnification'],  # Magnification of the tile
#         #     'tile_position': {
#         #         'region_y': 0,  # Adjust this as needed
#         #     },
#         #     # Other relevant tile information
#         # }
#         # tile['tile'].fill(meta['magnification'])
#         averageSize = opts.superpixelSize ** 2
#         overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0
#         tileSize = opts.tileSize + overlap


        
#         print('>> Reading input images')
#         # print(opts.inputImageFile)
#         try:
            
#             # meta = input_np.getMetadata(?)
#             pass
            
#         except Exception as inst:
#             print(type(inst))    # the exception type


#         # ts = large_image.open(opts.inputImageFile)
#         # meta = ts.getMetadata()

#         try:
#             found = 0
#             strips = []
#             bboxes = []
#             bboxesUser = []
#             tiparams = {}
            
#             #*******************temp*****************************
#             tiparams_1 = utils.get_region_dict(opts.roi, None, ts)
#             #*******************temp*****************************
#             single_tile_metadata = {
#                 'sizeX': input_np.shape[1],  # Width of the tile
#                 'sizeY': input_np.shape[0],  # Height of the tile
#             }
#             tiparams = utils.get_region_dict(opts.roi, None, single_tile_metadata)

#             # Use the metadata dictionary for the single tile in tiparams
            
            
            
#         except Exception as inst:
#             print(type(inst))    # the exception type

#         # ts = large_image.open(opts.inputImageFile)
#         print("tiparams",tiparams)
        
#         #*******************temp*****************************
        
#         for tile in ts.tileIterator(
#         format=large_image.constants.TILE_FORMAT_NUMPY,
#         tile_size=dict(width=tileSize, height=tileSize),
#         tile_overlap=dict(x=overlap, y=overlap),
#         **tiparams,
#     ):
#             if hasattr(opts, 'callback'):
#                 opts.callback('tiles', tile['tile_position']['position'],
#                             tile['iterator_range']['position'])
#             print('%d/%d (%d x %d) - %d' % (
#                 tile['tile_position']['position'], tile['iterator_range']['position'],
#                 tile['width'], tile['height'],
#                 found))
#             if meta['magnification'] and tile['magnification']:
#                 scale = meta['magnification'] / tile['magnification']
#             x0 = tiparams.get('region', {}).get('left', 0)
#             y0 = tiparams.get('region', {}).get('top', 0)
#             # tx0 = tile['x'] - x0
#             # ty0 = tile['y'] - y0
#             tx0 = int((tile['gx'] - x0) / scale)
#             ty0 = int((tile['gy'] - y0) / scale)
#             img = tile['tile']
#             n_pixels = tile['width'] * tile['height']
#             mask = None
#             if overlap:
#                 mask = numpy.ones(img.shape[:2])
#                 for y, simg in strips:
#                     if (y < ty0 + tile['height'] and y + simg.height > ty0 and simg.width > tx0):
#                         suby = max(0, y - ty0)
#                         subimg = simg.crop(
#                             tx0,
#                             max(0, ty0 - y),
#                             min(tile['width'], simg.width),
#                             min(tile['height'], simg.height - max(0, ty0 - y)))
#                         # Our mask is true when a pixel has not been set
#                         submask = numpy.ndarray(
#                             buffer=subimg[3].write_to_memory(),
#                             dtype=numpy.uint8,
#                             shape=[subimg.height, subimg.width]) == 0
#                         mask[suby:suby + submask.shape[0], :submask.shape[1]] *= submask
#                 n_pixels = numpy.count_nonzero(mask)
#             n_segments = math.ceil(n_pixels / averageSize)
#             if tile['tile_position']['position'] > 1:
#                 break
#         #*******************temp*****************************

#         # Get the region dictionary from utils.get_region_dict
#         region_dict = utils.get_region_dict(opts.roi, None, tile)
        

#         # Extract relevant information from region_dict
#         x_offset = region_dict.get('region', {}).get('left', 0)
#         y_offset = region_dict.get('region', {}).get('top', 0)



#         print("createSuperPixels_histomicstk started")
#         if meta['magnification'] and tile['magnification']:
#             scale = meta['magnification'] / tile['magnification']
#         # scale = 1
#         if opts.magnification:
#             tiparams['scale'] = {'magnification': opts.magnification}
#         x0 = tiparams.get('region', {}).get('left', 0)
#         y0 = tiparams.get('region', {}).get('top', 0)
#         # tx0 = tile['x'] - x0
#         # ty0 = tile['y'] - y0
#         # tx0 = int((tile['gx'] - x0) / scale)
#         # ty0 = int((tile['gy'] - y0) / scale)                        
            
            
#         img = tile['tile']  # The single tile you have computed
#         tileWidth = tile['width']
#         tileHeight = tile['height']
#         tileOverlapBottom = tile.get('tile_overlap', {}).get('bottom', 0)
#         tileOverlapRight = tile.get('tile_overlap', {}).get('right', 0)
#         tileRegionY = tile['tile_position'].get('region_y', 0)


#         print('>> Generating superpixels')
#         if opts.slic_zero:
#             print('>> Using SLIC Zero for segmentation')
            
#         mask = None
#         mask = numpy.ones(img.shape[:2])    
#         for y, simg in strips:
#             if (y < ty0 + tile['height'] and y + simg.height > ty0 and simg.width > tx0):
#                 suby = max(0, y - ty0)
#                 subimg = simg.crop(
#                     tx0,
#                     max(0, ty0 - y),
#                     min(tile['width'], simg.width),
#                     min(tile['height'], simg.height - max(0, ty0 - y)))
#                 # Our mask is true when a pixel has not been set
#                 submask = numpy.ndarray(
#                     buffer=subimg[3].write_to_memory(),
#                     dtype=numpy.uint8,
#                     shape=[subimg.height, subimg.width]) == 0
#                 mask[suby:suby + submask.shape[0], :submask.shape[1]] *= submask
#         n_pixels = numpy.count_nonzero(mask)
        
        
        
#         n_pixels = numpy.count_nonzero(mask)
#         n_segments = math.ceil(n_pixels / averageSize)
#         cuda_segments_slic = cuda_slic(img, n_segments=n_segments, max_iter=5)
#         return cuda_segments_slic
#         # segments = skimage.segmentation(
#         #     img,
#         #     n_segments=n_segments,
#         #     slic_zero=bool(opts.slic_zero),
#         #     compactness=opts.compactness,
#         #     sigma=opts.sigma,
#         #     start_label=0,
#         #     enforce_connectivity=True,
#         #     mask=mask,
#         # )
                            
                        
  
#                         # print('%d/%d (%d x %d) - %d' % (
#                         # 1, 1, tileWidth, tileHeight, found))  # Only one tile, so fixed values
                        
#                         # if meta['magnification'] and tile['magnification']:
#                         #     scale = meta['magnification'] / tile['magnification']
#                         # x0 = tiparams.get('region', {}).get('left', 0)
#                         # y0 = tiparams.get('region', {}).get('top', 0)
#                         # tx0 = int((tile['gx'] - x0) / scale)
#                         # ty0 = int((tile['gy'] - y0) / scale)
#                         # n_pixels = tileWidth * tileHeight
#                         # mask = None
 
#                         # if overlap:
#                         #     mask = np.ones(img.shape[:2])
#                         #     for y, simg in strips:
#                         #         if (y < ty0 + tile['height'] and y + simg.height > ty0 and simg.width > tx0):
#                         #             suby = max(0, y - ty0)
#                         #             subimg = simg.crop(
#                         #                 tx0,
#                         #                 max(0, ty0 - y),
#                         #                 min(tile['width'], simg.width),
#                         #                 min(tile['height'], simg.height - max(0, ty0 - y)))
#                         #             # Our mask is true when a pixel has not been set
#                         #             submask = numpy.ndarray(
#                         #                 buffer=subimg[3].write_to_memory(),
#                         #                 dtype=numpy.uint8,
#                         #                 shape=[subimg.height, subimg.width]) == 0
#                         #             mask[suby:suby + submask.shape[0], :submask.shape[1]] *= submask
#                         #     n_pixels = numpy.count_nonzero(mask)
#                         # n_segments = math.ceil(n_pixels / averageSize)
                    
                    
#                         # #  Iterate through input data and perform superpixel segmentation

                         
#                         # segments = skimage.segmentation.cuda_slic(
#                         # img,
#                         # n_segments=n_segments,
#                         # slic_zero=bool(opts.slic_zero),
#                         # compactness=opts.compactness,
#                         # sigma=opts.sigma,
#                         # start_label=0,
#                         # enforce_connectivity=True,
#                         # mask=mask,
#                         # )
#     #************************************** Overlap ************************************************************                  

#         maxValue = numpy.max(segments) + 1
#         if overlap:
#             # Keep any segment that is at all in the non-overlap region
#             core = segments[
#                 :tileHeight - tileOverlapBottom,
#                 :tileWidth - tileOverlapRight]
#             coremask = mask[
#                 :tileHeight - tileOverlapBottom,
#                 :tileWidth - tileOverlapRight]
#             core[numpy.where(coremask != 1)] = -1
#             usedIndices = numpy.unique(core)
#             usedIndices = numpy.delete(usedIndices, numpy.where(usedIndices < 0))
#             usedLut = [-1] * maxValue
#             for idx, used in enumerate(usedIndices):
#                 if used >= 0:
#                     usedLut[used] = idx
#             usedLut = numpy.array(usedLut, dtype=int)
#             print('reduced from %d to %d' % (maxValue, len(usedIndices)))
#             maxValue = len(usedIndices)
#             segments = usedLut[segments]
#             mask *= (segments != -1)

#         if str(opts.bounding).lower() not in {'', 'none'}:
#             regions = skimage.measure.regionprops(1 + segments)
#             for pidx, props in enumerate(regions):
#                 by0, bx0, by1, bx1 = props.bbox
#                 bboxes.append((
#                     ((bx0 + bx1) / 2 + tx0) * scale + x0,
#                     ((by0 + by1) / 2 + ty0) * scale + y0,
#                     (bx1 - bx0) * scale,
#                     (by1 - by0) * scale))
#                 bboxesUser.extend([
#                     (bx0 + tx0) * scale + x0,
#                     (by0 + ty0) * scale + y0,
#                     (bx1 + tx0) * scale + x0,
#                     (by1 + ty0) * scale + y0,
#                 ])
#         if opts.boundaries:
#             segments *= 2
#             maxValue *= 2
#             edges = (scipy.ndimage.sobel(segments, axis=0) != 0) | (
#                 scipy.ndimage.sobel(segments, axis=1) != 0)
#             edges[0, :] = True
#             edges[-1, :] = True
#             edges[:, 0] = True
#             edges[:, -1] = True
#             segments += edges
#         segments += found
#         found += int(maxValue)
#         if mask is None:
#             data = np.dstack((
#                 (segments % 256).astype(int),
#                 (segments / 256).astype(int) % 256,
#                 (segments / 65536).astype(int) % 256)).astype('B')
#         else:
#             data = np.dstack((
#                 (segments % 256).astype(int),
#                 (segments / 256).astype(int) % 256,
#                 (segments / 65536).astype(int) % 256,
#                 mask * 255)).astype('B')
#     # For overlay, suppose we make any value whose centroid is in the
#     # overlap region transparent.  Then, use vips to overlap the
#     # images rather than inserting them.
#         vimg = pyvips.Image.new_from_memory(
#             numpy.ascontiguousarray(data).data,
#             data.shape[1], data.shape[0], data.shape[2],
#             large_image.constants.dtypeToGValue[data.dtype.char])
#         vimg = vimg.copy(interpretation=pyvips.Interpretation.RGB)
#         vimgTemp = pyvips.Image.new_temp_file('%s.v')
#         vimg.write(vimgTemp)
#         vimg = vimgTemp
#         x = tx0
#         ty = tile['tile_position']['region_y']
#         while len(strips) <= ty:
#             strips.append(None)
#         if strips[ty] is None:
#             strip = pyvips.Image.black(
#                 tiparams.get('region', {}).get('width', meta['sizeX']),
#                 vimg.height, bands=vimg.bands)
#             strip = strip.copy(interpretation=pyvips.Interpretation.RGB)
#             strips[ty] = [ty0, strip]
#         strips[ty][1] = strips[ty][1].composite([vimg], pyvips.BlendMode.OVER, x=int(x), y=0)
#         # if hasattr(opts, 'callback'):
#         #     opts.callback('tiles', tile['tile_position']['position'] + 1,
#         #                     tile['iterator_range']['position'])
        
#         if hasattr(opts, 'callback'):
#             opts.callback('tiles', 1, 1)  # Only one tile, so fixed values
            
#     #    overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0
#     # bboxes = []
#     # bboxesUser = []
    

#         if hasattr(opts, 'callback'):
#             opts.callback('file', 0, 2 if opts.outputAnnotationFile else 1)
#         print('>> Found %d superpixels' % found)
#         if found > 256 ** 3:
#             print('Too many superpixels')
#             img = pyvips.Image.black(
#                 tiparams.get('region', {}).get('width', tileWidth) / scale,
#                 tiparams.get('region', {}).get('height', tileHeight) / scale,
#                 bands=strips[0][1].bands)
#             img = img.copy(interpretation=pyvips.Interpretation.RGB)
#             for stripidx in range(len(strips)):
#                 img = img.composite(
#                     [strips[stripidx][1]], pyvips.BlendMode.OVER, x=0, y=int(strips[stripidx][0]))
#             # Discard alpha band, if any.
#             img = img[:3]
#             # Add program run parameters to the image description and list the
#             # superpixel count
#             img.set_type(
#                 pyvips.GValue.gstr_type, 'image-description',
#                 json.dumps(dict(
#                     {k: v for k, v in vars(opts).items() if k != 'callback'}, indexCount=found)))
#             img.write_to_file(
#                 os.path.join(opts.outputImageFile, 'superpixel.tiff'),
#                 tile=True, tile_width=256, tile_height=256, pyramid=True,
#                 region_shrink=pyvips.RegionShrink.NEAREST,
#                 bigtiff=True, compression='lzw', predictor='horizontal')
            
#             print("createSuperPixels_histomicstk finished")
#             annotation_name = "annotation"
#             categories = [
#                 {
#                     'label': opts.default_category_label,
#                     'fillColor': opts.default_fillColor,
#                     'strokeColor': opts.default_strokeColor,
#                 },
#             ]
#             # Annotations
#             annotation = {
#                 'name': annotation_name,
#                 'elements': [{
#                     'type': 'pixelmap',
#                     'girderId': 'outputImageFile',
#                     'transform': {
#                         'xoffset': region_dict.get('region', {}).get('left', 0) / scale,
#                         'yoffset': region_dict.get('region', {}).get('top', 0) / scale,
#                         'matrix': [[scale, 0], [0, scale]],
#                     },
#                     'values': [0] * (found // (2 if opts.boundaries else 1)),
#                     'categories': categories,
#                     'boundaries': opts.boundaries,
#                 }],
#                 'attributes': {
#                     'params': vars(opts),
#                     'cli': Path(__file__).stem,
#                     'version': histomicstk.__version__,
#                 },
#             }
#             if len(bboxes) and str(opts.bounding).lower() != 'separate':
#                 annotation['elements'][0]['user'] = {'bbox': bboxesUser}
#             if len(bboxes) and str(opts.bounding).lower() != 'internal':
#                 bboxannotation = {
#                     'name': '%s bounding boxes' % os.path.splitext(
#                         os.path.basename(opts.outputAnnotationFile))[0],
#                     'elements': [{
#                         'type': 'rectangle',
#                         'center': [bcx, bcy, 0],
#                         'width': bw,
#                         'height': bh,
#                         'rotation': 0,
#                         'label': {'value': 'Region %d' % bidx},
#                         'fillColor': 'rgba(0,0,0,0)',
#                         'lineColor': opts.default_strokeColor,
#                     } for bidx, (bcx, bcy, bw, bh) in enumerate(bboxes)],
#                     'attributes': {
#                         'params': vars(opts),
#                         'cli': Path(__file__).stem,
#                         'version': histomicstk.__version__,
#                     },
#                 }
#                 annotation = [annotation, bboxannotation]
#             with open(opts.outputAnnotationFile, 'w') as annotation_file:
#                 json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)
#             if hasattr(opts, 'callback'):
#                 opts.callback('file', 2, 2)

#         overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0
#         bboxes = []
#         bboxesUser = []

#         if hasattr(opts, 'callback'):
#             opts.callback('file', 0, 2 if opts.outputAnnotationFile else 1)                       


#             segments = 1  # temp variable
#             return segments





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


def main():
    # Load large image using openslide from a file path
    image_path = "/tf/notebook/TCGA-TM-A7CF-01Z-00-DX1.EB905EDB-5AC5-41C4-AB00-526DD3820524.svs"
    slide_path = ["/tf/notebook/TCGA-TM-A7CF-01Z-00-DX1.EB905EDB-5AC5-41C4-AB00-526DD3820524.svs"]
    tempdir = "/tf/notebook/"
    outImagePath1 = "outputImageFile.tiff"
    annotationDir = "/tf/notebook/"
    annotationFolderId = "/tf/notebook/outputAnnotationFile.anot"
    outImagePath = os.path.join(tempdir, 'superpixel.tiff')
    annotationName = "Superpixel"
    outAnnotationPath = os.path.join(tempdir, '%s.anot' % annotationName)
    model_name = "superpixel_overlap"
    tiles_size = (4096, 4096)
    tile_size = 4096
    RUN = False
    
    ts = large_image.open(image_path)
    meta = ts.getMetadata()
    print("meta",meta)
    
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
    bounding='Internal',
    slic_zero=True,
    compactness=0.1,
    sigma=1,
    default_category_label='default',
    default_fillColor='rgba(0, 0, 0, 0)',
    default_strokeColor='rgba(0, 0, 0, 1)',
    )
    
    if RUN:
    # Loop through each batch
        for batch_idx in range(batch_size):
            # Loop through each tile in the batch
            for tile_idx in range(slide.level_dimensions[0][1] // tile_shape[0]):
                # Get the current tile from the slide
                tile = slide.read_region(
                    location=(0, tile_idx * tile_shape[0]),
                    level=0,
                    size=(tile_shape[1], tile_shape[0])
                )
                tile_np = np.array(tile)
                
                # segment= createSuperPixels_histomicstk(spopts, tile_np,image_path)
                print("segment", segment)
                n_segments = 125
                cuda_segments = cuda_slic(tile_np, n_segments=n_segments, max_iter=5)
                print("segment", cuda_segments)
                # Preprocessing: Perform any required preprocessing steps here
                # For example, you can apply color normalization, contrast enhancement, etc.
                # ...

                # Define parameters for SLIC segmentation
    n_segments = 100
    compactness = 10
    sigma = 1.0
    


    # Create Triton HTTP client
    triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    triton_client_grpc = grpcclient.InferenceServerClient(url="localhost:8001", verbose=True)

    # client_nogpu()
    # cache_clear()
    # gpu_mem_clear()


    # Load model
    try:
        triton_client.unload_model(model_name)
        triton_client.load_model(model_name)
        # create_load_model(model_name,"localhost:8001")
        print(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"Failed to load model '{model_name}': {e}")


    ################################################################################

    # warmup_model()

    ################################################################################

    # Create a histomic-stream study
    print("Create a histomic-stream study")
    hs_study = create_hs_study(slide_path, tile_size, tile_size, 20, "exact")

    outputs = [grpcclient.InferRequestedOutput("output")]
    output = [httpclient.InferRequestedOutput("output")]
     

    # Perform custom histomics stream inference
    print("Perform custom histomics stream inference")
    url="localhost:8001",
    batch=1,
    workers=32,
    limit=10


    num_iterations = 1
    average_throughput_tiles_cuda = 0
    
     # createSuperPixels_histomicstk(spopts,segment,tiles)

    for _ in range(num_iterations):
        start_time = time.time()
        response, tile_info, times, failed = histomics_stream_inference(hs_study, model_name, url="localhost:8001", batch=1, workers=32, limit=10)
        # response = triton_client_grpc.infer(model_name, inputs=inputs0, outputs=outputs)
        print("response:{} tile_info", response)
        elapsed_time = time.time() - start_time
        throughput_tiles_cuda = total_tiles / elapsed_time
        average_throughput_tiles_cuda += throughput_tiles_cuda
    average_throughput_tiles_cuda /= num_iterations
    
   
    print("            ----------------------------------            ")
    print("Average Throughput for CUDA-SLICin terms of tiles/sec:", average_throughput_tiles_cuda, "tiles/second")
    print("Average elapsed_time for CUDA-SLICin terms of seconds:", elapsed_time, "seconds")
    
    # Create a dictionary to map tile IDs to tuples of tile data and metadata
    tile_data_map = {}
    ts = large_image.open(image_path)
    meta = ts.getMetadata()
    print("meta",meta)
    createSuperPixels(spopts, response, tile_info, meta)
    if (RUN):
        def iterate_tiles(response, tile_info):
            tile_data_list = response[0]  # Extract the list of tile arrays from response
            for i, tile_data in enumerate(tile_data_list):
                
                
                meta_data = {
                    'tile_height': tile_info['tile_height'][i],
                    'tile_width': tile_info['tile_width'][i],
                    'overlap_height': tile_info['overlap_height'][i],
                    'overlap_width': tile_info['overlap_width'][i],
                    'filename': tile_info['filename'][i],
                    'slide_name': tile_info['slide_name'][i],
                    # ... other keys ...
                    'chunk_top': tile_info['chunk_top'][i],
                    'chunk_left': tile_info['chunk_left'][i],
                    'chunk_bottom': tile_info['chunk_bottom'][i],
                    'chunk_right': tile_info['chunk_right'][i],
                    'tile_top': tile_info['tile_top'][i],
                    'tile_left': tile_info['tile_left'][i]
                }
                
                tile_position = {
                    'top': meta_data['chunk_top'],
                    'left': meta_data['chunk_left'],
                    'bottom': meta_data['chunk_bottom'],
                    'right': meta_data['chunk_right']
                }
                
                yield {
                    'tile_data': tile_data,
                    'meta_data': meta_data,
                    'tile_position': tile_position
                }
                
                        
        for tile_info_dict in iterate_tiles(response, tile_info):
            tile_data = tile_info_dict['tile_data']
            meta_data = tile_info_dict['meta_data']
            tile_position = tile_info_dict['tile_position']
            tile_position = tile_info_dict['tile_position']
            
            # print("Tile Data:", tile_data)
            print("Meta Data:", meta_data)
            print("Tile Position:", tile_position)
            print("-----")
        
        # createSuperPixels_histomicstk(spopts, tile_data, meta_data)
    
    # createSuperPixels(spopts, tile_data, meta_data)
    
    
   
            
    print("FINISHED Inference")
                    # Perform SLIC segmentation
                    # segments = perform_slic_segmentation(tile_np, n_segments, compactness, sigma)

                    # # Define parameters for seam overlapping
                    # overlap_amount = 10
                    # strips = []  # Define 'strips' based on your requirements

                    # # If overlap is present, create a mask
                    # if overlap_amount:
                    #     # Define and populate the 'strips' variable based on your requirements
                    #     # strips = ...
                    #     mask = np.ones(tile_shape[:2])
                    #     for y, simg in strips:
                    #         if (y < tile_shape[0] and y + simg.height > 0 and simg.width > 0):
                    #             suby = max(0, y)
                    #             subimg = simg.crop(0, max(0, -y), min(tile_shape[1], simg.width), min(tile_shape[0], simg.height - max(0, -y)))
                    #             submask = np.array(subimg[3]) == 0
                    #             mask[suby:suby + submask.shape[0], :submask.shape[1]] *= submask
                    # else:
                    #     mask = None

                    # # Adjust the scaling factor based on magnification
                    # scale = slide.level_downsamples[0]

                    # # Perform seam overlapping
                    # overlapped_segments = perform_seam_overlapping(segments, overlap_amount)

                    # # Call the createSuperPixels function
                    # superpixel_opts = argparse.Namespace(
                    #     inputImageFile=image_path,
                    #     superpixelSize=tile_shape[0],  # You might need to adjust this based on your requirements
                    #     overlap=True,
                    #     tileSize=tile_shape[0],  # You might need to adjust this based on your requirements
                    #     magnification=None,  # Adjust if needed
                    #     slic_zero=True,
                    #     compactness=compactness,
                    #     sigma=sigma,
                    #     callback=None  # Replace with your callback function if needed
                    # )
                    # # createSuperPixels(superpixel_opts)

                    # # Display the current tile and the overlapped segmented tile
                    # plt.figure(figsize=(10, 5))

                    # plt.subplot(1, 2, 1)
                    # plt.imshow(tile_np)
                    # plt.title("Original Tile")
                    # plt.axis("off")

                    # plt.subplot(1, 2, 2)
                    # plt.imshow(overlapped_segments, cmap="nipy_spectral", interpolation="nearest")
                    # plt.title("Overlapped Segmented Tile")
                    # plt.axis("off")

                    # plt.tight_layout()
                    # plt.show()

            # Close the OpenSlide object
    slide.close()


if __name__ == "__main__":
    main()









# import numpy as np
# import skimage.segmentation
# import matplotlib.pyplot as plt
# import openslide

# def perform_slic_segmentation(image, n_segments, compactness, sigma):
#     segments = skimage.segmentation.slic(
#         image,
#         n_segments=n_segments,
#         compactness=compactness,
#         sigma=sigma,
#         start_label=0,
#         enforce_connectivity=True
#     )
#     return segments

# def perform_seam_overlapping(segments, overlap_amount):
#     overlapped_segments = segments.copy()
#     for i in range(overlap_amount):
#         overlapped_segments[:, i] = segments[:, 0]
#     return overlapped_segments

# def main():
#     # Load large image using openslide from a file path
#     image_path = "/tf/notebook/TCGA-05-4425-01Z-00-DX1.82B093EE-49BC-4FD9-91AC-4CC89944309D.svs"  # Replace with your image path
#     slide = openslide.OpenSlide(image_path)

#     # Get tile shape and batch size
#     tile_shape = (4096, 4096, 3)
#     batch_size = 1

#     # Loop through each batch
#     for batch_idx in range(batch_size):
#         # Loop through each tile in the batch
#         for tile_idx in range(slide.level_dimensions[0][1] // tile_shape[0]):
#             # Get the current tile from the slide
#             tile = slide.read_region(
#                 location=(0, tile_idx * tile_shape[0]),
#                 level=0,
#                 size=(tile_shape[1], tile_shape[0])
#             )
#             tile_np = np.array(tile)
#             # Define parameters for SLIC segmentation
#             n_segments = 100
#             compactness = 10
#             sigma = 1.0

#             # Perform SLIC segmentation
#             segments = perform_slic_segmentation(tile_np, n_segments, compactness, sigma)

#             # Define parameters for seam overlapping
#             overlap_amount = 10

#             # Perform seam overlapping
#             overlapped_segments = perform_seam_overlapping(segments, overlap_amount)

#             # Display the current tile and the overlapped segmented tile
#             plt.figure(figsize=(10, 5))

#             plt.subplot(1, 2, 1)
#             plt.imshow(tile_np)
#             plt.title("Original Tile")
#             plt.axis("off")

#             plt.subplot(1, 2, 2)
#             plt.imshow(overlapped_segments, cmap="nipy_spectral", interpolation="nearest")
#             plt.title("Overlapped Segmented Tile")
#             plt.axis("off")

#             plt.tight_layout()
#             plt.show()

#     # Close the OpenSlide object
#     slide.close()

# if __name__ == "__main__":
#     main()
