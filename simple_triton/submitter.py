import multiprocessing
from multiprocessing import Process, Queue
import numpy as np
import time


"""The submitter maintains a list of inference requests. It pulls samples from
the queue 'qin' and tracks these in 'requests'. A random number from [1-10]
is generated for each request that is decremented each time the request is 
checked. The inference result is served when this value reaches 0. On
completion the input data (a unique int) and time elapsed is printed. The
completed request is then placed in the output queue 'qout'.

The producer kills the running processes by inserting None values into 
'qin'.

Relevance to HistomicsML: the producer will read feature files (.tfr or 
.npy) from disk and insert these into qin. The consumer will make inference
requests to triton and collect and return the results to the producer
process.

Why I picked this design: 
1. If we have significant pre- or post-processing work, or if submitting 
requests has significant overhead, we can parallelize with multiple submitters. 
2. The template functions give a lot of flexibility for different inference 
server protocols or preprocessing tasks. 
3. We can use this approach to solve a lot of problems besides HistomicsML and
doing inference with .tfr files. For example, extracting features from images
to generate .tfr files can be done with this method. Here, the producer
would read tiles from an image and use triton to infer features for each tile.
4. We can also stack these submitters if we have sequential inference tasks. 
For example, say we have 2 models, one for cell detection and the other for 
cell classification. We can have a submitter to the cell detection model and 
then feed its output/inferences to the classifier model input queue.

Todo:
    -Write put/get functions for GRPC and system shared memory
        -How to pass args to put/get, pre/post function calls
    -Error checking, timeouts, logging, etc.
    -Integrate with class
    -Add throughput / performance tracker
        -Track time spent: preprocessing, waiting for requests, 
         postprocessing
    -Dynamically adjust sleep period and request queue length
"""


class Submitter(Process):
    
    def __init__(self, qin, qout, limit, put, get, pre, post):
        multiprocessing.Process.__init__(self)
        self.qin = qin # the input queue that holds data to do inference on
        self.qout = qout # the output queue that holds inference results
        self.limit = limit # the limit on number of outstanding inference requests 
        self.put = put # the function to make an inference request
        self.get = get # the function to retrieve an inference request
        self.pre = pre # an optional preprocessing function to apply to data before inference - e.g. color normalization
        self.post = post # an optional postprocessing function to apply to completed inference results
        
    def run(self):

        # initialize list of pending inference requests 
        requests = []
        
        # set flag indicating qin stop signal receipt
        stop = False
        
        # loop until exit signal received from calling process
        while True:
            
            print(f"{self.name} - {len(requests)} requests")
            
            # if pending requests < limit, pull sample from input queue
            if(len(requests) < self.limit) and not stop:
                
                # pull sample from producer
                sample = self.qin.get()
            
                # check stop signal
                if sample is None:
                    stop = True
                    continue
                
                # apply preprocessing function
                sample = self.pre(sample)
            
                # append request to list
                requests.append(self.put(sample))
                
            # check pending requests
            delete = []
            for i in range(len(requests)):
                result = self.get(requests[i])
                if result is not None:
                    result = self.post(result)
                    self.qout.put(result)
                    delete.append(i)
    
            # delete completed requests
            delete.reverse()
            for i in delete:
                _ = requests.pop(i)
                
            # check if done
            if len(requests) == 0 and stop:
                break
                    
            # sleep
            time.sleep(0.1)
            
        return


# function for submitting request to triton
# each request holds a "sample" (a unique identifier), a random value that is
# decremented each time the request is checked, and the time the request was
# started. When decremented to 0, the request is finished.
def put_dummy(sample):
    request = {"sample": sample,
               "served": np.random.randint(low=1, high=10),
               "start": time.time()}
    return request
    

# function for checking if request is completed - should be non-blocking 
# so that other requests can be checked if one is not ready
def get_dummy(request):
    
    # decrement dummy request value
    request["served"] = request["served"]-1
    
    # check if inference resullt is served
    if request["served"] == 0:
        request["elapsed"] = time.time() - request["start"]
        return request
    else:
        return None


# preprocessing function - applied to data prior to inference
# ex. color normalization
def pre_dummy(sample):
    return sample


# postprocessing function - applied to inference result prior to returning
# here we just print the "inference" result
def post_dummy(inference):
    print_inference(inference)
    return inference


# print request
def print_inference(inference):
    print("Sample {}: {:0.3} seconds".format(inference["sample"],
                                             inference["elapsed"]),
          flush=True)


if __name__ == '__main__':
    
    # parameters
    N = 100 # total number of inferences to perform
    tasks = list(range(N)) # the input sample to each "inference" is just an int
    limit = 5 # limit on number of pending requests per worker
    workers = 4 # total number of Submitter workers
    
    # start timer
    start = time.time()
    
    # create input, output queues
    qin = Queue()
    qout = Queue()
    
    # Start consumers
    print(f"Creating {workers} workers")
    consumers = [Submitter(qin,
                           qout,
                           limit,
                           put_dummy,
                           get_dummy,
                           pre_dummy,
                           post_dummy)
                 for i in range(workers)]
    for w in consumers:
        w.start()

    # enqueue tasks
    print("Enqueuing inference jobs")
    for i in tasks:
        qin.put(i)
    
    # enqueue stop signals
    for i in range(workers):
        qin.put(None)

    # collecct results
    print("Collecting results")
    results = []
    while N:
        results.append(qout.get())
        N -= 1
    for result in results:
        print_inference(result)

    # display elapsed time
    print(f"Total elapsed time: {time.time()-start}")
