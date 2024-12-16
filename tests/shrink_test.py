import onnx
from onnx import helper as h, TensorProto as tp
import onnxruntime as ort
import numpy as np

from onnx_shrink_ray.shrink import quantize_weights

def check_quantization(float_model):
    onnx.checker.check_model(float_model)

    quantized_model = quantize_weights(float_model, min_elements=2)
    quantized_float_model = quantize_weights(float_model, min_elements=2, float_quantization=True)

    float_session = ort.InferenceSession(float_model.SerializeToString())
    actual_float_output = np.array(
        float_session.run(None, {})[0], 
        dtype=np.float32)

    quantized_session = ort.InferenceSession(quantized_model.SerializeToString())
    actual_quantized_output = np.array(
        quantized_session.run(None, {})[0], 
        dtype=np.float32)

    quantized_float_session = ort.InferenceSession(quantized_float_model.SerializeToString())
    actual_quantized_float_output = np.array(
        quantized_float_session.run(None, {})[0], 
        dtype=np.float32)

    output_min = np.min(actual_float_output)
    output_max = np.max(actual_float_output)
    output_range = output_max - output_min
    output_bin_size = output_range / 255.0
    
    # print(f"actual_float_output: {actual_float_output}")
    # print(f"actual_quantized_output: {actual_quantized_output}")
    
    output_diff = np.abs(actual_float_output - actual_quantized_output)
    max_diff = np.max(output_diff)
    if max_diff > output_bin_size:
        raise Exception(f"Max difference {max_diff} is greater than output bin size {output_bin_size}.")

    output_qf_diff = np.abs(actual_float_output - actual_quantized_float_output)
    max_qf_diff = np.max(output_qf_diff)
    if max_qf_diff > output_bin_size:
        raise Exception(f"Max difference {max_qf_diff} is greater than output bin size {output_bin_size}.")

def test_single_constant():
    weights_shape = (1, 1, 2, 2)
    weights_values = np.array([[[[0.0, 2.5], [5.0, 10.0]]]], dtype=np.float32)

    weights_tensor = h.make_tensor(name="weights_tensor", data_type=tp.FLOAT,
        dims=weights_shape,
        vals=weights_values)

    weights_node = h.make_node("Constant", inputs=[], outputs=["weights_output"], name="weights_node",
        value=weights_tensor)

    float_graph = h.make_graph([weights_node], "test_graph",
        [],
        [h.make_tensor_value_info("weights_output", tp.FLOAT, weights_shape)])

    float_model = h.make_model(float_graph, producer_name="quantization_test")

    check_quantization(float_model)

def test_identity():
    weights_shape = (1, 1, 2, 2)
    weights_values = np.array([[[[0.0, 2.5], [5.0, 10.0]]]], dtype=np.float32)

    weights_tensor = h.make_tensor(name="weights_tensor", data_type=tp.FLOAT,
        dims=weights_shape,
        vals=weights_values)

    weights_node = h.make_node("Constant", inputs=[], outputs=["weights_output"], name="weights_node",
        value=weights_tensor)

    identity_node = h.make_node("Identity", inputs=["weights_output"], outputs=["identity_output"], name="identity_node")

    float_graph = h.make_graph([weights_node, identity_node], "test_graph",
        [],
        [h.make_tensor_value_info("identity_output", tp.FLOAT, weights_shape)])

    float_model = h.make_model(float_graph, producer_name="quantization_test")

    check_quantization(float_model)

def test_mul():
    weights_shape = (1, 1, 2, 2)
    weights_values = np.array([[[[0.0, 2.5], [5.0, 10.0]]]], dtype=np.float32)
    weights_tensor = h.make_tensor(name="weights_tensor", data_type=tp.FLOAT,
        dims=weights_shape,
        vals=weights_values)
    weights_node = h.make_node("Constant", inputs=[], outputs=["weights_output"], name="weights_node",
        value=weights_tensor)

    two_shape = (1, )
    two_values = np.array([2.0], dtype=np.float32)
    two_tensor = h.make_tensor(name="two_tensor", data_type=tp.FLOAT,
        dims=two_shape,
        vals=two_values)
    two_node = h.make_node("Constant", inputs=[], outputs=["two_output"], name="two_node",
        value=two_tensor)

    mul_node = h.make_node("Mul", inputs=["weights_output", "two_output"], outputs=["mul_output"], name="mul_node")

    float_graph = h.make_graph([weights_node, two_node, mul_node], "test_graph",
        [],
        [h.make_tensor_value_info("mul_output", tp.FLOAT, weights_shape)])

    float_model = h.make_model(float_graph, producer_name="quantization_test")

    check_quantization(float_model)

def test_large_constant():
    weights_width = 256
    weights_height = 256
    weights_shape = (1, 1, weights_height, weights_width)
    rng = np.random.default_rng(7528840384)
    weights_values = rng.random((weights_shape)).astype(np.float32)

    weights_tensor = h.make_tensor(name="weights_tensor", data_type=tp.FLOAT,
        dims=weights_shape,
        vals=weights_values)

    weights_node = h.make_node("Constant", inputs=[], outputs=["weights_output"], name="weights_node",
        value=weights_tensor)

    float_graph = h.make_graph([weights_node], "test_graph",
        [],
        [h.make_tensor_value_info("weights_output", tp.FLOAT, weights_shape)])

    float_model = h.make_model(float_graph, producer_name="quantization_test")

    check_quantization(float_model)

def test_signed_constant():
    weights_shape = (1, 1, 2, 2)
    weights_values = np.array([[[[-5.0, -2.5], [0.0, 5.0]]]], dtype=np.float32)

    weights_tensor = h.make_tensor(name="weights_tensor", data_type=tp.FLOAT,
        dims=weights_shape,
        vals=weights_values)

    weights_node = h.make_node("Constant", inputs=[], outputs=["weights_output"], name="weights_node",
        value=weights_tensor)

    float_graph = h.make_graph([weights_node], "test_graph",
        [],
        [h.make_tensor_value_info("weights_output", tp.FLOAT, weights_shape)])

    float_model = h.make_model(float_graph, producer_name="quantization_test")

    check_quantization(float_model)


def test_unbalanced_constant():
    weights_shape = (1, 1, 2, 2)
    weights_values = np.array([[[[-2.0, 0.5], [3.0, 8.0]]]], dtype=np.float32)

    weights_tensor = h.make_tensor(name="weights_tensor", data_type=tp.FLOAT,
        dims=weights_shape,
        vals=weights_values)

    weights_node = h.make_node("Constant", inputs=[], outputs=["weights_output"], name="weights_node",
        value=weights_tensor)

    float_graph = h.make_graph([weights_node], "test_graph",
        [],
        [h.make_tensor_value_info("weights_output", tp.FLOAT, weights_shape)])

    float_model = h.make_model(float_graph, producer_name="quantization_test")

    check_quantization(float_model)

if __name__ == "__main__":
    test_single_constant()
    test_identity()
    test_mul()
    test_large_constant()
    test_signed_constant()
    test_unbalanced_constant()
    print("All tests passed.")
