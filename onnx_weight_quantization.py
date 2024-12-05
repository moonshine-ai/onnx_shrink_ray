import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Don't quantify constants smaller than this.
DEFAULT_MIN_ELEMENTS = 16 * 1024

def quantize_tensor(name, value_tensor, original_output_tensor_name, graph):
    float_values = value_tensor.values
    min_val = np.min(float_values)
    max_val = np.max(float_values)
    range_val = max_val - min_val
    inverse_range = 1.0 / range_val
    zero_point = round(-min_val * inverse_range * 255.0) - 128
    # y = (x - zero_point) * scale
    # print(f"Min: {min_val}, Max: {max_val}, Range: {range_val}, Inverse Range: {inverse_range}, Zero Point: {zero_point}")
    quantized_values = np.round(float_values * inverse_range * 255.0) + zero_point
    # print(f"Quantized values: {quantized_values}")
    quantized_values = np.clip(quantized_values, -128, 127).astype(np.int8)

    quantized_tensor = gs.Constant(
        name=f"{name}_quantized", 
        values=quantized_values)
    
    zero_point_tensor = gs.Constant(
        name=f"{name}_zero_point", 
        values=np.array([zero_point], dtype=np.int8))

    scale_value = range_val / 255.0        
    scale_tensor = gs.Constant(
        name=f"{name}_scale",
        values=np.array([scale_value], dtype=np.float32))
    
    dequantized_tensor_name = f"{name}_dequantized_tensor"
    dequantized_tensor = gs.Variable(
        name=dequantized_tensor_name, 
        dtype=np.float32,
        shape=value_tensor.shape)

    dequantized_node = gs.Node(
        op="DequantizeLinear", 
        name=f"{name}_dequantized_node", 
        inputs=[quantized_tensor, scale_tensor, zero_point_tensor],
        outputs=[dequantized_tensor])

    for node in graph.nodes:
        for i, tensor in enumerate(node.inputs):
            if tensor.name == original_output_tensor_name:
                node.inputs[i] = dequantized_tensor

    for i, tensor in enumerate(graph.outputs):
        if tensor.name == original_output_tensor_name:
            graph.outputs[i] = dequantized_tensor

    # Add the quantized tensor to the graph.
    graph.nodes.append(dequantized_node)

def float_quantize_node(name, value_tensor, original_output_tensor_name, graph, levels=256):
    float_values = value_tensor.values
    min_val = np.min(float_values)
    max_val = np.max(float_values)
    range_val = max_val - min_val
    inverse_range = 1.0 / range_val
    zero_point = round(-min_val * inverse_range * 255.0) - 128
    scale_value = range_val / 255.0        
    # y = (x - zero_point) * scale
    # print(f"Min: {min_val}, Max: {max_val}, Range: {range_val}, Inverse Range: {inverse_range}, Zero Point: {zero_point}")
    quantized_values = np.round(float_values * inverse_range * 255.0) + zero_point
    # print(f"Quantized values: {quantized_values}")
    quantized_values = np.clip(quantized_values, -128, 127).astype(np.int8)
    dequantized_values = ((quantized_values.astype(np.int32) - zero_point) * scale_value).astype(np.float32)
    # print(f"Dequantized values: {dequantized_values}")

    dequantized_tensor = gs.Constant(
        name=f"{name}_dequantized", 
        values=dequantized_values)    

    for node in graph.nodes:
        for i, tensor in enumerate(node.inputs):
            if tensor.name == original_output_tensor_name:
                node.inputs[i] = dequantized_tensor

    for i, tensor in enumerate(graph.outputs):
        if tensor.name == original_output_tensor_name:
            graph.outputs[i] = dequantized_tensor

    # Add the quantized tensor to the graph.
    # graph.nodes.append(dequantized_tensor)

def quantize_weights(model, min_elements=DEFAULT_MIN_ELEMENTS, float_quantization=False):
    graph = gs.import_onnx(model)

    original_graph = graph.copy()

    # Quantize the weights
    for node in original_graph.nodes:
        if node.op != "Constant":
            # print(f"Node {node.name} is not a Constant node. Skipping quantization.")
            continue
        name = node.name
        value_tensor = node.attrs["value"]
        original_output_tensor_name = node.outputs[0].name
        elements = np.prod(value_tensor.shape)
        if elements < min_elements:
            # print(f"Not quantizing {name} with {elements} elements.")
            continue
        if float_quantization:
            float_quantize_node(name, value_tensor, original_output_tensor_name, graph)
        else:
            quantize_tensor(name, value_tensor, original_output_tensor_name, graph)

    for name, value_tensor in original_graph.tensors().items():
        if value_tensor.__class__ != gs.Constant:
            continue
        original_output_tensor_name = name
        elements = np.prod(value_tensor.shape)
        if elements < min_elements:
            # print(f"Not quantizing {name} with {elements} elements.")
            continue
        if float_quantization:           
            float_quantize_node(name, value_tensor, original_output_tensor_name, graph)
        else:
            quantize_tensor(name, value_tensor, original_output_tensor_name, graph)
    
    graph.cleanup(remove_unused_graph_inputs=False).toposort()

    no_shape_model = gs.export_onnx(graph)
    new_model = onnx.shape_inference.infer_shapes(no_shape_model)

    onnx.checker.check_model(new_model)
    
    return new_model


if __name__ == "__main__":
    import glob
    import os
    import sys

    if len(sys.argv) < 2:
        input_globs = ["*.onnx"]
    else:
        all_args = sys.argv[1:]
        input_globs = []
        float_quantization = False
        for arg in all_args:
            if arg.startswith("-"):
                if arg == "-f" or arg == "--float":
                    float_quantization = True
                else:
                    print(f"Unknown option: {arg}")
                    sys.exit(1)
            else:
                input_globs.append(arg)

    for input_glob in input_globs:
        input_filenames = list(glob.glob(input_glob))
        if len(input_filenames) == 0:
            print(f"No files found matching '{input_glob}'.")
            sys.exit(1)

        for input_filename in input_filenames:
            if input_filename.endswith("_quantized_weights.onnx"):
                print(f"Skipping {input_filename} as it is already quantized.")
                continue
            original_model = onnx.load(input_filename)
            new_model = quantize_weights(original_model, float_quantization=float_quantization)
            output_filename = os.path.splitext(input_filename)[0] + "_quantized_weights.onnx"
            onnx.save(new_model, output_filename)
