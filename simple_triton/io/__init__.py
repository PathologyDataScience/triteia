"""
This subpackage contains functions for reading, writing, and transforming data
"""

# slide level keys from histomics_stream study
slide_keys = [
    "filename",
    "slide_name",
    "slide_group",
    "chunk_height",
    "chunk_width",
    "target_magnification",
    "scan_magnification",
    "read_magnification",
    "returned_magnification",
    "level",
    "slide_height",
    "slide_width",
    "tile_height",
    "tile_width",
    "overlap_height",
    "overlap_width",
    "slide_height_tiles",
    "slide_width_tiles",
    "mask_height",
    "mask_width",
]

# tile level keys from histomics_stream
tile_keys = ["chunk_left", "chunk_top", "tile_left", "tile_top", "slide_index"]

# make functions available at the package level using shadow imports
from .tfr_reader import peek, read_record
from .tfr_transforms import flatten, parallel_dataset, structure
from .tfr_writer import merge_records, write_labels, write_record

# list out things that are available for public use
__all__ = (
    "merge_records",
    "parallel_dataset",
    "peek",
    "read_record",
    "structure",
    "write_labels",
    "write_record",
)
