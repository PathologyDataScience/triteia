from builtins import range
from functools import partial
import multiprocessing
from multiprocessing import Process, Queue
import numpy as np
import requests
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

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

class SimulatedProducer(object):
    """A simulated producer that emits numpy arrays with specified batch size
    and feature dimensions.
    
    Data is uniformly distributed and so compression ratio will be low.
    """
    
    def __init__(self, B=1024, D=1024, dtype=np.float16):
        """Constructor.
        
        Parameters
        ----------
        B : int
            Batch size. Default value is 1024.
        D : int
            Feature dimension. Default value is 1024.
        dtype : numpy.dtype
            A numpy dtype for the emited data. Default value is float16.
        """
        
        self.B = B # batch size
        self.D = D # dimension
        self.dttype = dtype # datatype as float16 or float32
        

    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        output = self.dtype(np.random.uniform(size=(self.B, self.D)))

        return output


class Requests(object):
    """A class to manage inference server requests.
    
    The class maintains a list of pending requests, ordered by submission time.
    It can be used to check on the status of pending requests and to insert
    new requests.
    """
     
    def __init__(self, client, limit):
        """Construct
        
        Parameters
        ----------
        client : tritonclient.grpc object
            A GRPC client used to submit requests. 
        limit : int
            The 
        """
        
        self.client = client
        self.limit = limit
        self.pending = []

    
    def _api_types(self, dtype):
        """Converts numpy dtype to triton API type string.
        
        Parameters
        ----------
        dtype : numpy.dtype
            A numpy dtype.
        
        Returns
        -------
        api_type : str
            The corresponding type string for the triton client API.
        """
        
        if dtype == np.float32:
            return "FP32"
        elif dtype == np.float16:
            return "FP16"
        elif dtype == np.float64:
            return "FLOAT64"
        elif dtype == np.uint8:
            return "UINT8"
        elif dtype == np.uint16:
            return "UINT16"
        elif dtype == np.uint32:
            return "UINT32"
        elif dtype == np.uint64:
            return "UINT64"
        elif dtype == np.int8:
            return "INT8"
        elif dtype == np.int16:
            return "INT16"
        elif dtype == np.int32:
            return "INT32"
        elif dtype == np.int64:
            return "INT64"
        elif dtype == np.bool:
            return "BOOL"
        else:
            raise ValueError(f"Unrecognized type '{str(dtype)}'")
            
    
    def _callback(self, capture, result, error):
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
            capture.append(error)
        else:
            capture.append(result)
        
    
    def check(self, block=True, wait=100e-3):
        """Check for completion of pending requests.
        
        Parameters
        ----------
        block : bool
            If True, the function will block until at least one request 
            completes. Default value is True.
        wait : float
            If block is True, this is the interval to wait until checking
            requests again.
        
        Returns
        -------
        completed : list of tuple
            A list containing the results of completed inference requests.
        """
        
        # iterate through list of requests, checking who is finished
        def request_loop():
            
            # initialize completed requests
            completed = []
            
            # initalize list of requests to delete
            delete = []
            
            for i, request in enumerate(requests):
                if len(request['result']):
                    
                    # record elapsed time
                    request['request_elapsed'] = time.time() - request['request_elapsed']
                    
                    # add request to output list
                    completed.append(request)
                    
                    # add request to list for deletion
                    delete.append(i)
                    
            return completed, delete
                    
        # if no blocking, iterate through list once and return
        if not block:   
            completed, delete = request_loop()
        else:
            while True:
                completed, delete = request_loop()
                if len(completed):
                    break
                time.sleep(wait)
        
        # delete completed entries from list
        self.pending.reverse()
        for i in delete:
            _ = self.pending.pop(i)
        self.pending.reverse()
        
        return completed
    
    
    def insert(self, model_name, sample, timeout=None):
        """Insert an inference request for submission to triton.
        
        Parameters
        ----------
        model_name : string
            The name of a served model.
        sample : list or tuple of list, dict
            If list, contains a numpy array for each of the model inputs.
            If tuple, the first element is the list of input arrays, and the
            second is the sample metadata dictionary that will stay attached
            to the inference result.
        timeout : float
            Timeout for the request in seconds. Default value is None.
        """
        
        # process input sample - if list or tuple
        if isinstance(sample, tuple):
            data, metadata = sample
        elif isinstance(sample, list):
            data = sample
            metadata = None
        
        # create InputData objects based on data shape
        inputs = []
        for i, k in enumerate(data):
            ii = grpcclient.InferInput(f"INPUT{k}",
                                       list(i.shape),
                                       self._api_types(i.dtype))
            ii.set_data_from_numpy(i)
            inputs.append(ii)

        # create outputs
        outputs = [grpcclient.InferRequestedOutput('OUTPUT0')]
        
        # create a dict to hold timing information, metadata, and request
        # completion
        request = {'metadata': metadata,
                   'data': data,
                   'result': [],
                   'exception': None,
                   'request_elapsed': time.time()}
        
        # submit request
        self.client.async_infer(model_name=model_name,
                                inputs=inputs,
                                callback=partial(self._callback, 
                                                 request['result']),
                                outputs=outputs,
                                client_timeout=timeout)
        
        # append request to list
        self.pending.append(request)


class InferenceRunner(Process):
    """InferenceRunner """
    
    def __init__(self, 
                 url, 
                 qin, 
                 qout,
                 limit=10,
                 rest=1e-2,
                 timeout=None,
                 pre=None, 
                 post=None, 
                 verbose=False):
        """InferenceRunner constructor.
        
        Parameters
        ----------
        url : string
            The inference server url.
        qin : multiprocessing.Queue
            Input queue containing samples for inference.
        qout : multiprocessing.Queue
            Output queue receiving completed inference requests and 
            InferenceRunner process performance information.
        limit : int
            The maximum allowable pending requests. Default value is 10.
        rest : float
            The resting period for the InferenceRunner process. The process
            will rest for this period (seconds) after submitting and checking
            inference requests. Default value is 10 milliseconds.
        timeout : float
            The request timeout limit (seconds). This is an input argument
            to the triton GRPC client async_infer inferface.
        pre : function
            A preprocessing function to apply to samples prior to inference.
            Default value is None.
        post : function
            A postprocessing function to apply to inference results. Default
            value is None.
        verbose : bool
            True updates console with inference progress and exceptions. 
            Default value is False.
        """
        
        multiprocessing.Process.__init__(self)
        
        # create GRPC client
        try:
            self.client = grpcclient.InferenceServerClient(url=url,
                                                           verbose=verbose)
        except Exception as e:
            print("context creation failed: " + str(e), flush=True)
            return
        
        # capture input arguments
        self.qin = qin
        self.qout = qout
        self.limit = limit
        self.timeout = timeout
        self.rest = rest
        self.pre = pre
        self.post = post
        
        # initialize list to hold performance data
        self.time_inference = []
    

    def run(self):

        # set flag indicating qin stop signal received
        stop = False
                
        # create requests object
        req = Requests(self.client, self.limit)

        # loop until exit signal received from calling process
        while True:
            
            # fill input queue with requests up to limit
            if not stop:
                for _ in range(self.limit - len(req.pending)):
                    
                    # pull sample
                    sample = self.qin.get()
                    
                    # check if stop signal
                    if sample is None:
                        stop = True
                        break
                    
                    # apply preprocessing function
                    # TBD

                    # fill requests if stop signal not received
                    req.insert(sample, self.timeout)

            # check pending requests
            completed = req.check(block=False)
            
            # put completed post-processed requests into queue
            for result in completed:
                
                # check if 
                if type(result) == InferenceServerException:
                
                    # add sample to retry
                    # TBD
                    print(result, flush=True)
                
                else:
                    
                    # capture performance data
                    self.time_inference.append(result["request_elapsed"])
                
                    # apply post processing function
                    # TBD
                    
                    # place in queue
                    self.qout.put((result['result'], result['metadata']))
            
            # check if done
            if len(req.pending) == 0 and stop:
                break

            # sleep
            time.sleep(self.rest)
            
        return


#function for incrementing the count of input infer/output request position
def put_pos():
	global count
	count  += 1
	print(count)
	return
# function for submitting request to triton
# each request holds a "sample" (a unique identifier), a random value that is
# decremented each time the request is checked, and the time the request was
# started. When decremented to 0, the request is finished.

def put_dummy(sample):
     
    results_data = []   
    print(count)
    
    # Adding the while to handle inference exception in the start
    # the below condition will tell the loop to stop
    while True:

    #break the loop when limit has reached.
     if (count == N-1):
        break

    # Inference call
     triton_grpc_client.async_infer(model_name, inputs=[input0[count]],
                                    callback=partial(callback, results_data),
                                        outputs=[output[count]])

    # Wait until the results are available in results_data
     time_out = 10
     while ((len(results_data) == 0) and time_out > 0):
            time_out = time_out - 1
            time.sleep(0.1)

     try:
    # Display and validate the available results
    # To handle InferenceServerException, checking the values in first two
    # places. This code can be optimized later to reduce the if statements

      if ((len(results_data) == 0)):
        # Check for the errors
        if type(results_data[0]) == InferenceServerException:
            print(results_data[0])
            # output0_data = 0
      
     
      if len(results_data)>=1:
        if type(results_data[0]) == InferenceServerException:
            print(results_data[0])
            # output0_data = 0
        else:
            # Validate the values by matching with already computed expected values.
            output0_data = results_data[len(results_data)-1].as_numpy('output_0')
            # print the received values for verification
            # print(len(output0_data))
            # for i in range(len(output0_data)):
            #  print(output0_data[i])
            request = {"sample": sample,
                    "served": np.random.randint(low=1, high=10),
                    "start": time.time(),
                    "response": output0_data}
            put_pos()
            break
            

     except :
      print("pass")
      pass 

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
    count = 0 # postion of input inference and out request in the list
    tasks = list(range(N)) # the input sample to each "inference" is just an int
    limit = 5 # limit on number of pending requests per worker
    workers = 4 # total number of Submitter workers
    grpc_url = 'localhost:8001' # url for grpc access to tirton server
    model_version = '1' # set model version
    verbose = False # set verbos as False
    input_dtype = 'FP16' # set input data type
    model_name = 'simple-trt-model-FP16' # set model name
    model_name_test = 'simple-trt-model-FP16-test' # set model name
    input_name = 'input_0' # set input name
    output_name = 'output_0' # set putput name
    model_path_input='models/simple-trt-model-FP16-input/1/model.savedmodel' # set model path
    model_path_test='models/simple-trt-model-FP16-test/1/model.savedmodel' # set model path
    batch_size=1024
    input0 = [i for i in range(N)]
    output = [i for i in range(N)]
    
    
     
        # check connectivity with triton server and model
    res = requests.get('http://localhost:8000/v2/health/ready')
    print(f"Tirton Server connection status: {res}")
    res = requests.get('http://localhost:8000/v2/models/simple-trt-model-FP16-test')
    print(f"Model connection status: {res}")
    
    # create tirton grpc client: gRPC is a newer, open source remote 
    # procedure call system initially developed at Google in 2015 that
    # uses HTTP/2 for transport and Protocol Buffers as the interface 
    # description language. It is highly efficient.
    triton_grpc_client = grpcclient.InferenceServerClient(url=grpc_url, verbose=verbose)


   # instantiate triton client using the tritonhttpclient.InferenceServerClient class
   #  access the model metadata with the .get_model_metadata() method as well as get 
   # our model configuration with the get_model_config() method.
    client = grpcclient.InferenceServerClient(url=grpc_url, verbose=verbose)
    model_metadata = client.get_model_metadata(model_name=model_name, model_version=model_version)
    model_config = client.get_model_config(model_name=model_name, model_version=model_version)

    for x in range(N):

     # Generate the InferInput and corresponding InferRequestedOutput
    # for the Triton Inference Server here. Values are then pass on 
    # to put_dummy() for inference.
    # Alternatively we can generate the input/output pair at put_dummy()
    # This part of code can be moved to a sperate function
    # It can be modifed after discussion
        batch1 = (SimulatedProducer(batch_size))
        i = iter(batch1)
        batch=next(i)

        # use the tritonclient.grpc module to instantiate new InferInput and InferRequestedOutput objects
        input0[x] = grpcclient.InferInput(input_name, batch.shape, 'FP16')
        input0[x].set_data_from_numpy(batch)
        output[x] = grpcclient.InferRequestedOutput(output_name)


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
