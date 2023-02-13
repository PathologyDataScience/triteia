from functools import partial
import numpy as np
import sys
import time
import tritonclient.grpc as grpcclient


def attend_model(event,
        url="localhost:8001",
        model_name="densenet_onnx",
        inputs={"name": "data_0", "datatype": "FP32", "shape": [1, 3, 224, 224]},
        outputs="fc6_1",
        interval = 10e-1):
    
    """A testing utility used to submit inference requests to a Triton server.
    
    This process submits events to a model with period `interval` seconds to
    make the model appear active. This active state is necessary for testing
    functions related to model loading / unloading. Default parameters
    correspond to the `densenet_onnx` model used in Triton examples, served 
    from a local instance.
    
    Parameters
    ----------
    event : multiprocessing.Event
        An event used by the main process to signal the experiment end.
    url : string
        The inference server url. Default value is "localhost:8001".
    model_name : string
        The name of the model to submit to. Default value is "densenet_onnx".
    inputs : dict
        A dictionary defining the name, type, and shape of model inputs.
        Limited to single input models. Default value is for `densenet_onnx`.
    outputs : str
        The name of the model output. Default value is for `densenet_onnx`.
    shape : list or tuple of int
        The shape of each sample to send in [Batch, C, W, H] format. 
        Default value is [1, 3, 224, 224].
    interval : float
        Request submission interval in seconds.
    """    
    
            
    # create client
    client = grpcclient.InferenceServerClient(url=url, verbose=False)
    
    # define inference callback
    def callback(user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)
            
    while True:
        
        # start timer
        start = time.time()
        
        # exit if testing process sets event (on experiment end)
        if event.is_set():
            sys.exit(1)
        
        # build input and output structures for next inference
        # do not reuse since triton output caching for repeated inputs is
        # not understood
        data = np.random.uniform(size=inputs["shape"]).astype(np.float32)
        infer_inputs = [grpcclient.InferInput(inputs["name"],
                                              inputs["shape"],
                                              inputs["datatype"])
                  ]
        infer_inputs[0].set_data_from_numpy(data)            
        infer_outputs = [grpcclient.InferRequestedOutput(outputs)]
        
        # infer
        discard = []
        client.async_infer(model_name=model_name,
                           inputs=infer_inputs,
                           callback=partial(callback, discard),
                           outputs=infer_outputs,
                           client_timeout=None)           
        
        # sleep until next inference
        time.sleep(max(0, interval-(time.time()-start)))
