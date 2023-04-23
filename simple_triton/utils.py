import tensorflow as tf
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def create_client(url="localhost:8001", vebose=False):
    """Create a grpcclient.
    
    Parameters
    ----------
    url : string
        The url for the remote-procedure call port of the Triton server.
        Default value is "localhost:8001".
    verbose : bool
        If True the client will emit status messages to stdout. Default
        is False.
    
    Returns
    -------
    client : grpcclient.InferenceServerClient
        A client 
    """

    try:
        self.client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
    except Exception as e:
        print("context creation failed: " + str(e), flush=True)
    return client


def reshape_savedmodel(
    path,
    savedmodel,
    shape,
    dtype=tf.float32,
    signature="serving_default",
    outputs=["output_1"],
):
    """Modify a savedmodel to change input signature.

    Some savedmodel files lack a batch dimension or have input
    dimensions set when they should be variable. This makes hosting
    on Triton difficult. This function generates a copy of the
    savedmodel with an altered input signature.

    Parameters
    ----------
    path : str
        The path for the output savedmodel.
    savedmodel : str
        The path to the savedmodel root folder.
    shape: array-like
        The new input shape signature. Use `None` for variable
        dimensions.
    dtype : tensorflow.python.framework.dtypes.DType
        The output dtype.
    signature : str
        The savedmodel signature to modify and save.
    outputs : array-like
        A list of outputs to capture in the modified savedmodel.

    Notes
    -----
    Models altered using this method may need to be loaded on
    triton with the platform explicitly mentioned in the
    configuration: {"platform": "tensorflow_savedmodel"}.
    """

    # define a class that discards batch dimension
    class Reshaped(tf.Module):
        def __init__(self, model):
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec(shape, dtype)])
        def fn(self, input_1):
            inference = self.model.signatures[signature](input_1[0])
            return {o: inference[o] for o in outputs}

    # instantiate and call model
    model = tf.saved_model.load(savedmodel)
    reshaped = Reshaped(model)
    tf.saved_model.save(
        reshaped,
        path,
        signatures={signature: reshaped.fn},
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=False),
    )
