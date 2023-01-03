from builtins import range
from functools import partial
import multiprocessing
from multiprocessing import Process, Queue
import numpy as np
import requests
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

"""
Todo:
    -Limit input size queue
    -Write put/get functions for GRPC and system shared memory
        -How to pass args to put/get, pre/post function calls
    -Error checking, timeouts, logging, etc.
    -Add throughput / performance tracker
        -Track time spent: preprocessing, waiting for requests, 
         postprocessing
    -Dynamically adjust sleep period and request queue length
"""


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
            Output queue receiving completed inference requests and process
            summary information on process completion.
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
     grpcclient.async_infer(model_name, inputs=[input0[count]],
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


if __name__ == '__main__':
    
    # parameters
    N = 100 # total number of inferences to perform
    count = 0 # postion of input inference and out request in the list
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
    model_path_input = 'models/simple-trt-model-FP16-input/1/model.savedmodel' # set model path
    model_path_test = 'models/simple-trt-model-FP16-test/1/model.savedmodel' # set model path
    batch_size = 1024
    dimension = 1024
    
    # check connectivity with triton server and model
    res = requests.get('http://localhost:8000/v2/health/ready')
    print(f"Tirton Server connection status: {res}")
    res = requests.get('http://localhost:8000/v2/models/simple-trt-model-FP16-test')
    print(f"Model connection status: {res}")

    # start timer
    start = time.time()
    
    # create input, output queues
    qin = Queue()
    qout = Queue()
    
    # initialize producer
    producer = iter(SimulatedProducer(batch_size, dimension, np.float16))
      
    # Start consumers
    print(f"Creating {workers} workers")
    consumers = [InferenceRunner(grpc_url,
                                 qin,
                                 qout,
                                 limit,
                                 verbose=verbose)
                 for _ in range(workers)]
    for w in consumers:
        w.start()

    # enqueue tasks
    print("Enqueuing inference jobs")
    for _ in range(N):
        data = next(producer)
        metadata = {'key': 'random stuff'}
        qin.put((model_name, data, metadata))
    
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
        print(result)

    # display elapsed time
    print(f"Total elapsed time: {time.time()-start}")
