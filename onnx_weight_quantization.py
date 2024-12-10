import onnx_graphsurgeon as gs
import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

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
    half_levels = (levels / 2)
    zero_point = round(-min_val * inverse_range * (levels - 1)) - half_levels
    scale_value = range_val / (levels - 1)
    quantized_values = np.round(float_values * inverse_range * (levels - 1)) + zero_point
    # print(f"Quantized values: {quantized_values}")
    quantized_values = np.clip(quantized_values, -half_levels, (half_levels - 1))
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

def quantize_weights(model, min_elements=DEFAULT_MIN_ELEMENTS, float_quantization=False, float_levels=256):
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
            float_quantize_node(name, value_tensor, original_output_tensor_name, graph, levels=float_levels)
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
            float_quantize_node(name, value_tensor, original_output_tensor_name, graph, levels=float_levels)
        else:
            quantize_tensor(name, value_tensor, original_output_tensor_name, graph)
    
    graph.cleanup(remove_unused_graph_inputs=False).toposort()

    no_shape_model = gs.export_onnx(graph)
    new_model = onnx.shape_inference.infer_shapes(no_shape_model)

    onnx.checker.check_model(new_model)
    
    return new_model


if __name__ == "__main__":
    import argparse
    import glob
    import os
    import sys

    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Quantization utility for ONNX models",
    )
    parser.add_argument(
        "--method", "-m",
        help="How to quantize the models",
        default="integer_weights",
        choices=["integer_weights", "float_weights", "integer_activations"],
    )
    parser.add_argument(
        "--float_levels", "-l",
        help="Number of levels to use for float quantization.",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Folder to write the quantized models to. If not specified, uses the same folder as the input models.",
        default=None,
    )
    parser.add_argument(
        "--output_suffix", "-s",
        help="Suffix to add to the output model filenames.",
        default="quantized_weights.onnx",
    )
    parser.add_argument("globs", nargs="*")
    args = parser.parse_args()
    if len(args.globs) == 0:
        args.globs = ["*.onnx"]

    if args.output_dir is not None and not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for input_glob in args.globs:
        if os.path.isdir(input_glob):
            input_glob = os.path.join(input_glob, "*.onnx")
        input_filenames = list(glob.glob(input_glob))
        if len(input_filenames) == 0:
            print(f"No files found matching '{input_glob}'.")
            sys.exit(1)

        for input_filename in input_filenames:
            if args.output_suffix != ".onnx" and input_filename.endswith(args.output_suffix):
                print(f"Skipping '{input_filename}' as it is already quantized.")
                continue
            input_base = os.path.basename(input_filename)
            input_dir = os.path.dirname(input_filename)
            output_base = os.path.splitext(input_base)[0] + args.output_suffix
            if args.output_dir is None:
                output_filename = os.path.join(input_dir, output_base)
            else:
                output_filename = os.path.join(args.output_dir, output_base)
            if output_filename == input_filename:
                print(f"Skipping '{input_filename}' as the output filename is the same and it would be overwritten.")
                continue
            if args.method == "float_weights" or args.method == "integer_weights":
                original_model = onnx.load(input_filename)
                float_quantization = (args.method == "float_weights")
                new_model = quantize_weights(original_model, float_quantization=float_quantization, float_levels=args.float_levels)
                onnx.save(new_model, output_filename)
            elif args.method == "integer_activations":
                quantize_dynamic(input_filename, output_filename, weight_type=QuantType.QUInt8)
            else:
                print(f"Unknown quantization method: {args.method}")
                sys.exit(1)
