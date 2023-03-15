class ShardedTiles(object):
    """Iterator for sharded reading with large image.
    
    This class is used for sharded multiprocessing reading using
    the large_image library. This can be used to improve throughput
    for inference tasks. The large_image object used for reading is
    created on the first read, making ShardedLargeImage object 
    serializable.
    
    Parameters
    ----------
    study : dict
        A study dictionary from histomics_stream, defining the reading
        parameters and tile locations for possibly multiple slides.
    batch : int
        The number of tiles in each batch. Partial batches are not padded.
    worker_index : int
        The worker index, ranging from 0 to `num_workers`.
    num_workers : int
        The total number of workers.
    
    Returns
    -------
    tiles : array-like
        A four-dimensional BHWC numpy array of batched tiles.
    metadata : dict
        TBD
    
    Attributes
    ----------
    batch : int
        The number of tiles in each batch. Partial batches are not padded.
    worker_index : int
        The worker index, ranging from 0 to `num_workers`.
    num_workers : int
        The total number of workers.

    Notes
    -----
    See https://github.com/DigitalSlideArchive/HistomicsStream/blob/master/StudyObject.md
    for more details on `metadata`.
    
    """
    
    def __init__(self, study, batch, worker_id, num_workers):
        self.study = study
        self.batch = batch
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.large_images = None
        self._shard()
    
    def _shard(self):
        # generates the large_image read parameters for this worker's shard
        reads = [(slide["filename"],
                  {"scale": {"magnification": slide["target_magnification"]},
                   "format": "numpy",
                   "region": {
                       "left": tile["tile_left"],
                       "top": tile["tile_top"],
                       "width": study["tile_width"],
                       "height": study["tile_height"],
                       "units": "mag_pixels"
                   }
                  }
                 )
                 for slide in self.study["slides"].values() 
                 for tile in slide["tiles"].values()]
        indices = range(len(reads) * self.worker_index // self.num_workers,
                        len(reads) * (self.worker_index + 1) // self.num_workers)
        self.tiles = [reads[i] for i in indices]
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= len(self.tiles):
            self.__iter__()
            raise StopIteration
        else:
            if self.large_images is None: # lazy creation of large_image objects enables serialization
                self.large_images = {self.tiles[self.i][0]: large_image.open(self.tiles[self.i][0])}
            else: 
                if self.tiles[self.i][0] not in self.large_images: # shards can span multiple slides - slide not open yet
                    self.large_images[self.tiles[self.i][0]] = large_image.open(self.tiles[self.i][0])
            indices = range(self.i, min(self.i+self.batch, len(self.tiles)))
            pixels = [self.large_images[self.tiles[j][0]].getRegion(**self.tiles[j][1])[0] for j in indices]
            metadata = [self.tiles[j][1] for j in indices]
            pixels = np.stack(pixels, axis=0)
            self.i += len(indices)
            return pixels, metadata
