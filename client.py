import numpy as np
import tritonclient.http as httpclient

model_name = "cuda_slic"
# Prepare dummy inputs
input_data_0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
input_data_1 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

# Create Triton HTTP client
triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)

# Create InferenceRequest
inputs = []
inputs.append(httpclient.InferInput("INPUT0", input_data_0.shape, "FP32"))
inputs.append(httpclient.InferInput("INPUT1", input_data_1.shape, "FP32"))
inputs[0].set_data_from_numpy(input_data_0)
inputs[1].set_data_from_numpy(input_data_1)

outputs = [
    httpclient.InferRequestedOutput("OUTPUT0"),
    httpclient.InferRequestedOutput("OUTPUT1"),
]

response = triton_client.infer(model_name,
                               inputs,
                               request_id=str(1),
                               outputs=outputs)
# Process and print the output
output_0 = response.as_numpy("OUTPUT0")
output_1 = response.as_numpy("OUTPUT1")
print("Input 0:", input_data_0)
print("Input 1:", input_data_1)
print("Output 0:", output_0)
print("Output 1:", output_1)
