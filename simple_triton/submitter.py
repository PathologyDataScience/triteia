import argparse
from builtins import range
from ctypes import *
import easydict
from functools import partial
import multiprocessing
from multiprocessing import Process, Queue
import numpy as np
import gevent.ssl
import requests
import sys
if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue
import time
from tqdm import tqdm
import tritonclient.grpc as grpcclient
from tritonclient import utils
from tritonclient.utils import InferenceServerException
import tritonclient.utils.shared_memory as shm
from tritonclient.utils import triton_to_np_dtype


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

# create numpy array for consumer to pass
class iterator(object):

    def __init__(self, B=2048, D=1024):
        self.B = B  # batch size
        self.D = D  # dimension

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        output = np.float16(np.random.uniform(size=(self.B, self.D)))

        return output


# create callable for asynchronous requests to triton server
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


class Submitter(Process):

    def __init__(self, qin, qout, limit, put, get, pre, post):
        multiprocessing.Process.__init__(self)
        self.qin = qin  # the input queue that holds data to do inference on
        self.qout = qout  # the output queue that holds inference results
        self.limit = limit  # the limit on number of outstanding inference requests
        self.put = put  # the function to make an inference request
        self.get = get  # the function to retrieve an inference request
        self.pre = pre  # an optional preprocessing function to apply to data before inference - e.g. color normalization
        # an optional postprocessing function to apply to completed inference results
        self.post = post

    def run(self):

        # initialize list of pending inference requests
        requests = []

        # initialize results for callback in async request for triton server
        results = []

        # set flag indicating qin stop signal receipt
        stop = False

        # loop until exit signal received from calling process
        while True:

            print(f"{self.name} - {len(requests)} requests")

            # if pending requests < limit, pull sample from input queue
            if (len(requests) < self.limit) and not stop:

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

    async_requests = []
    sent_count = 1
    user_data = UserData()
    async_requests.append(triton_grpc_client.async_infer(model_name, inputs=[input0],
                                                         callback=partial(
                                                             completion_callback, user_data),
                                                         outputs=[output]))

    # processed_count = 0
    # while processed_count < sent_count:
    #     (results, error) = user_data._completed_requests.get()
    #     processed_count += 1
    #     if error is not None:
    #         print("inference failed: " + str(error))
    #         sys.exit(1)
    #     responses.append(user_data)
    request = {"sample": sample,
               "served": np.random.randint(low=1, high=10),
               "start": time.time(),
               "response": async_requests}
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
# callback for async inference


def callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)


if __name__ == '__main__':

    # parameters
    N = 100  # total number of inferences to perform
    # the input sample to each "inference" is just an int
    tasks = list(range(N))
    limit = 5  # limit on number of pending requests per worker
    workers = 4  # total number of Submitter workers
    grpc_url = 'localhost:8001'  # url for grpc access to tirton server
    model_version = '1'  # set model version
    verbose = False  # set verbos as False
    input_dtype = 'FP16'  # set input data type
    model_name = 'simple-trt-model-FP16'  # set model name
    model_name_test = 'simple-trt-model-FP16-test'  # set model name
    input_name = 'input_0'  # set input name
    output_name = 'output_0'  # set putput name
    model_path_input = 'models/simple-trt-model-FP16-input/1/model.savedmodel'  # set model path
    model_path_test = 'models/simple-trt-model-FP16-test/1/model.savedmodel'  # set model path
    batch_size = 2048

    # check connectivity with triton server and model
    res = requests.get('http://localhost:8000/v2/health/ready')
    print(f"Tirton Server connection status: {res}")
    res = requests.get(
        'http://localhost:8000/v2/models/simple-trt-model-FP16-test')
    print(f"Model connection status: {res}")

    # create tirton grpc client: gRPC is a newer, open source remote
    # procedure call system initially developed at Google in 2015 that
    # uses HTTP/2 for transport and Protocol Buffers as the interface
    # description language. It is highly efficient.
    triton_grpc_client = grpcclient.InferenceServerClient(
        url=grpc_url, verbose=verbose)

   # instantiate triton client using the tritonhttpclient.InferenceServerClient class
   #  access the model metadata with the .get_model_metadata() method as well as get
   # our model configuration with the get_model_config() method.
    triton_grpc_client = grpcclient.InferenceServerClient(
        url=grpc_url, verbose=verbose)
    model_metadata = triton_grpc_client.get_model_metadata(
        model_name=model_name, model_version=model_version)
    model_config = triton_grpc_client.get_model_config(
        model_name=model_name, model_version=model_version)

    # input data for the Triton Inference Server
    batch1 = (iterator(batch_size))
    i = iter(batch1)
    batch = next(i)

    # use the tritonclient.grpc module to instantiate new InferInput and InferRequestedOutput objects
    input0 = grpcclient.InferInput(input_name, batch.shape, 'FP16')
    input0.set_data_from_numpy(batch)
    output = grpcclient.InferRequestedOutput(output_name)

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
