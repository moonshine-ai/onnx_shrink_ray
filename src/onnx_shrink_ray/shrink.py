import onnx_graphsurgeon as gs
import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


# Don't quantify constants smaller than this.
DEFAULT_MIN_ELEMENTS = 16 * 1024


def replace_tensor_for_subgraph(graph, original_tensor_name, new_tensor):
    """Replace a tensor in a graph with a new tensor.

    Args:
        graph: The graph to modify.
        original_tensor_name: The name of the tensor to replace.
        new_tensor: The tensor to replace it with.
    """
    for node in graph.nodes:
        for subgraph in node.attrs.values():
            if isinstance(subgraph, gs.Graph):
                replace_tensor_for_subgraph(subgraph, original_tensor_name, new_tensor)
        for i, tensor in enumerate(node.inputs):
            if tensor.name == original_tensor_name:
                node.inputs[i] = new_tensor

    for i, tensor in enumerate(graph.outputs):
        if tensor.name == original_tensor_name:
            graph.outputs[i] = new_tensor


def gather_initializers_in_graph(graph, all_initializers):
    for initializer in graph.initializer:
        all_initializers[initializer.name] = initializer

    graph.initializer.clear()

    for node in graph.node:
        if node.op_type == "If":
            for attr in node.attribute:
                if attr.name == "then_branch":
                    all_initializers = gather_initializers_in_graph(
                        attr.g, all_initializers
                    )
                elif attr.name == "else_branch":
                    all_initializers = gather_initializers_in_graph(
                        attr.g, all_initializers
                    )

    return all_initializers


def hoist_subgraph_initializers(onnx_model):
    """GraphSurgeon seems to leave duplicated initializers in the graph, so remove them."""

    all_initializers = {}
    gather_initializers_in_graph(onnx_model.graph, all_initializers)

    for name, initializer in all_initializers.items():
        onnx_model.graph.initializer.append(initializer)

    return onnx_model


def quantize_tensor(name, value_tensor, original_output_tensor_name, graph, root_graph):
    """Quantize a constant tensor to int8 using the DequantizeLinear op.

    Args:
        name: The name of the tensor to quantize.
        value_tensor: The tensor to quantize.
        original_output_tensor_name: The name of the original tensor in the graph.
        graph: The graph to modify.
        root_graph: The root graph of the model.
    """
    dtype = value_tensor.dtype
    np_dtype = onnx.helper.np_dtype_to_tensor_dtype(dtype)
    float_values = value_tensor.values
    min_val = np.min(float_values)
    max_val = np.max(float_values)
    range_val = max_val - min_val
    inverse_range = 1.0 / range_val
    zero_point = round(-min_val * inverse_range * 255.0) - 128
    quantized_values = np.round(float_values * inverse_range * 255.0) + zero_point
    quantized_values = np.clip(quantized_values, -128, 127).astype(np.int8)

    quantized_tensor = gs.Constant(name=f"{name}_quantized", values=quantized_values)

    scale_value = range_val / 255.0
    scale_tensor = gs.Constant(
        name=f"{name}_scale", values=np.array([scale_value], dtype=dtype)
    )

    zero_point_tensor = gs.Constant(
        name=f"{name}_zero_point",
        values=np.array([-zero_point * scale_value], dtype=dtype),
    )

    # DequantizeLinear is surprisingly slow in the OnnxRuntime, so achieve the
    # same effect with a Cast, Mul, and Add.
    cast_tensor_name = f"{name}_cast_tensor"
    cast_tensor = gs.Variable(
        name=cast_tensor_name, dtype=dtype, shape=value_tensor.shape
    )
    cast_node = gs.Node(
        op="Cast",
        name=f"{name}_cast_node",
        inputs=[quantized_tensor],
        outputs=[cast_tensor],
        attrs={"to": np_dtype},
    )

    mul_tensor_name = f"{name}_mul_tensor"
    mul_tensor = gs.Variable(
        name=mul_tensor_name, dtype=dtype, shape=value_tensor.shape
    )
    mul_node = gs.Node(
        op="Mul",
        name=f"{name}_mul_node",
        inputs=[cast_tensor, scale_tensor],
        outputs=[mul_tensor],
    )

    add_tensor_name = f"{name}_add_tensor"
    add_tensor = gs.Variable(
        name=add_tensor_name, dtype=dtype, shape=value_tensor.shape
    )
    add_node = gs.Node(
        op="Add",
        name=f"{name}_add_node",
        inputs=[mul_tensor, zero_point_tensor],
        outputs=[add_tensor],
    )

    replace_tensor_for_subgraph(root_graph, original_output_tensor_name, add_tensor)

    root_graph.nodes.append(cast_node)
    root_graph.nodes.append(mul_node)
    root_graph.nodes.append(add_node)


def float_quantize_node(
    name, value_tensor, original_output_tensor_name, root_graph, levels=256
):
    """Quantize a constant tensor to a small number of float values.

    Args:
        name: The name of the tensor to quantize.
        value_tensor: The tensor to quantize.
        original_output_tensor_name: The name of the original tensor in the graph.
        graph: The graph to modify.
        levels: The number of levels to quantize to.
    """
    dtype = value_tensor.dtype
    float_values = value_tensor.values
    min_val = np.min(float_values)
    max_val = np.max(float_values)
    range_val = max_val - min_val
    inverse_range = 1.0 / range_val
    half_levels = levels / 2
    zero_point = round(-min_val * inverse_range * (levels - 1)) - half_levels
    scale_value = range_val / (levels - 1)
    quantized_values = (
        np.round(float_values * inverse_range * (levels - 1)) + zero_point
    )
    quantized_values = np.clip(quantized_values, -half_levels, (half_levels - 1))
    dequantized_values = (
        (quantized_values.astype(np.int32) - zero_point) * scale_value
    ).astype(dtype)

    dequantized_tensor = gs.Constant(
        name=f"{name}_dequantized", values=dequantized_values
    )

    replace_tensor_for_subgraph(
        root_graph, original_output_tensor_name, dequantized_tensor
    )


def quantize_weights_for_graph(
    graph,
    root_graph,
    already_processed,
    min_elements=DEFAULT_MIN_ELEMENTS,
    float_quantization=False,
    float_levels=256,
    verbose=False,
):
    for node in graph.nodes:
        for subgraph in node.attrs.values():
            if isinstance(subgraph, gs.Graph):
                if verbose:
                    print(f"Processing subgraph {subgraph.name}")
                already_processed = quantize_weights_for_graph(
                    subgraph,
                    root_graph,
                    already_processed,
                    min_elements,
                    float_quantization,
                    float_levels,
                )
        if node.op != "Constant":
            continue
        name = node.name
        value_tensor = node.attrs["value"]
        if value_tensor.dtype not in [np.float16, np.float32, np.float64]:
            continue
        original_output_tensor_name = node.outputs[0].name
        if original_output_tensor_name in already_processed:
            continue
        already_processed.add(original_output_tensor_name)
        elements = np.prod(value_tensor.shape)
        if elements < min_elements:
            continue
        if verbose:
            print(f"Processing node {name}")
        if float_quantization:
            float_quantize_node(
                name,
                value_tensor,
                original_output_tensor_name,
                root_graph,
                levels=float_levels,
            )
        else:
            quantize_tensor(
                name, value_tensor, original_output_tensor_name, graph, root_graph
            )

    for name, value_tensor in graph.tensors().items():
        if value_tensor.dtype not in [np.float16, np.float32, np.float64]:
            continue
        if value_tensor.__class__ != gs.Constant:
            continue
        original_output_tensor_name = name
        if original_output_tensor_name in already_processed:
            continue
        already_processed.add(original_output_tensor_name)
        elements = np.prod(value_tensor.shape)
        if elements < min_elements:
            continue
        if verbose:
            print(f"Processing initializer {name}")
        if float_quantization:
            float_quantize_node(
                name,
                value_tensor,
                original_output_tensor_name,
                root_graph,
                levels=float_levels,
            )
        else:
            quantize_tensor(
                name, value_tensor, original_output_tensor_name, graph, root_graph
            )

    return already_processed


def quantize_weights(
    input_data,
    min_elements=DEFAULT_MIN_ELEMENTS,
    float_quantization=False,
    float_levels=256,
    verbose=False,
    ir_version=None,
):
    """Quantize the weights of an ONNX model.

    Args:
        input_data: The path or contents of the ONNX model to quantize.
        min_elements: The minimum number of elements a tensor must have to be quantized.
        float_quantization: If True, store the quantized values as float, not integers.
        float_levels: The number of levels to quantize to if using float quantization.
        verbose: If True, log detailed information about the weight processing.
        ir_version: The IR version to use for the output ONNX files.
    """
    if verbose:
        print(
            f"quantize_weights(input_data, min_elements={min_elements}, float_quantization={float_quantization}, float_levels={float_levels})"
        )

    graph = gs.import_onnx(input_data)

    already_processed = set()
    quantize_weights_for_graph(
        graph,
        graph,
        already_processed,
        min_elements,
        float_quantization,
        float_levels,
        verbose,
    )

    graph.cleanup(remove_unused_graph_inputs=False).toposort(recurse_subgraphs=True)

    no_shape_model = gs.export_onnx(graph)
    deduped_model = hoist_subgraph_initializers(no_shape_model)
    new_model = onnx.shape_inference.infer_shapes(deduped_model)
    if ir_version is not None:
        new_model.ir_version = ir_version

    onnx.checker.check_model(new_model)

    return new_model


def print_weight_info_for_graph(
    onnx_graph,
    total_bytes,
    node_count,
    initializer_count,
    already_processed,
    min_elements=DEFAULT_MIN_ELEMENTS,
):
    for node in onnx_graph.node:
        value_tensor = None
        for attribute in node.attribute:
            if attribute.name == "value":
                value_tensor = attribute.t
            if attribute.HasField("g"):
                subgraph = attribute.g
                total_bytes, node_count, initializer_count, already_processed = (
                    print_weight_info_for_graph(
                        subgraph,
                        total_bytes,
                        node_count,
                        initializer_count,
                        already_processed,
                        min_elements,
                    )
                )
        if node.op_type != "Constant":
            continue
        output_tensor_name = node.output[0]
        if output_tensor_name in already_processed:
            continue
        already_processed.add(output_tensor_name)
        name = node.name
        elements = np.prod(value_tensor.dims)
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(value_tensor.data_type)
        byte_count = int(elements * np_dtype.itemsize)
        total_bytes += byte_count
        if elements < min_elements:
            continue
        node_count += 1
        print(
            f"Node: {name}: {value_tensor.dims} - {elements} elements, {np_dtype}, {byte_count:,} bytes"
        )

    duplicate_names = set()
    for value_tensor in onnx_graph.initializer:
        name = value_tensor.name
        if name in already_processed:
            duplicate_names.add(name)
            continue
        already_processed.add(name)
        elements = np.prod(value_tensor.dims)
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(value_tensor.data_type)
        byte_count = int(elements * np_dtype.itemsize)
        total_bytes += byte_count
        if elements < min_elements:
            continue
        initializer_count += 1
        print(
            f"Initializer: {name}: {value_tensor.dims} - {elements:,} elements, {np_dtype}, {byte_count:,} bytes"
        )

    if len(duplicate_names) > 0:
        print(f"Duplicate initializers: {duplicate_names}")

    return total_bytes, node_count, initializer_count, already_processed


def print_weight_info(filename_or_model, min_elements=DEFAULT_MIN_ELEMENTS):
    """Return information about the size of the weights in an ONNX model.

    Args:
        model: The ONNX model to inspect.
    """
    if isinstance(filename_or_model, str):
        filename = filename_or_model
        onnx_model = onnx.load(filename)
        file_byte_count = os.path.getsize(filename)
        print(f"Model: {filename}")
    else:
        onnx_model = filename_or_model
        file_byte_count = onnx_model.ByteSize()

    total_bytes = 0
    node_count = 0
    initializer_count = 0
    already_processed = set()

    total_bytes, node_count, initializer_count, already_processed = (
        print_weight_info_for_graph(
            onnx_model.graph,
            total_bytes,
            node_count,
            initializer_count,
            already_processed,
            min_elements,
        )
    )

    print(f"Total nodes: {node_count}")
    print(f"Total initializers: {initializer_count}")
    print(
        f"Total bytes from weights: {total_bytes:,} bytes, {file_byte_count - total_bytes:,} bytes from other data"
    )
    print("-------------------------------------------")


def convert_f16_to_f32_tensor(
    name, value_tensor, original_output_tensor_name, root_graph, verbose=False
):
    """Quantize a constant tensor to a small number of float values.

    Args:
        name: The name of the tensor to quantize.
        value_tensor: The tensor to quantize.
        original_output_tensor_name: The name of the original tensor in the graph.
        graph: The graph to modify.
    """
    dtype = value_tensor.dtype
    assert dtype == np.float16
    f16_values = value_tensor.values
    f32_values = f16_values.astype(np.float32)
    f32_tensor = gs.Constant(name=f"{name}_f32", values=f32_values)

    replace_tensor_for_subgraph(root_graph, original_output_tensor_name, f32_tensor)


def convert_f16_to_f32(input_model, verbose=False, ir_version=None, nodes_to_exclude=None):
    graph = gs.import_onnx(input_model)

    already_processed = set()
    convert_f16_to_f32_for_graph(graph, graph, already_processed, verbose, nodes_to_exclude)

    graph.cleanup(remove_unused_graph_inputs=False).toposort(recurse_subgraphs=True)

    # no_shape_model = gs.export_onnx(graph)
    # deduped_model = hoist_subgraph_initializers(no_shape_model)
    # new_model = onnx.shape_inference.infer_shapes(deduped_model)

    new_model = gs.export_onnx(graph, do_type_check=False)

    if ir_version is not None:
        new_model.ir_version = ir_version

    return new_model


def convert_f16_to_f32_for_graph(graph, root_graph, already_processed, verbose=False, nodes_to_exclude=None):
    for node in graph.nodes:
        for subgraph in node.attrs.values():
            if isinstance(subgraph, gs.Graph):
                if verbose:
                    print(f"Processing subgraph {subgraph.name}")
                already_processed = convert_f16_to_f32_for_graph(
                    subgraph, root_graph, already_processed, verbose, nodes_to_exclude
                )
        if node.op == "Cast":
            if node.attrs["to"] == 10 or node.attrs["to"] == 1:
                if nodes_to_exclude is not None and node.name in nodes_to_exclude:
                    if verbose:
                        print(f"Changing cast {node.name} to float32 because it is in nodes_to_exclude")
                    node.attrs["to"] = 1
                else:
                    if verbose:
                        print(f"Removing float16 to float 32 cast ({node.name})")
                    cast_input_tensor = node.inputs[0]
                    cast_input_tensor.dtype = np.float32
                    replace_tensor_for_subgraph(
                        root_graph, node.outputs[0].name, cast_input_tensor
                    )
        if node.op != "Constant":
            continue
        name = node.name
        value_tensor = node.attrs["value"]
        if value_tensor.dtype != np.float16:
            if verbose:
                print(
                    f"Skipping Constant '{name}' because it is not float16 ({value_tensor.dtype})"
                )
            continue
        original_output_tensor_name = node.outputs[0].name
        if original_output_tensor_name in already_processed:
            continue
        already_processed.add(original_output_tensor_name)
        if verbose:
            print(f"Processing Constant {name}")
        convert_f16_to_f32_tensor(
            name, value_tensor, original_output_tensor_name, graph, verbose
        )

    for name, value_tensor in graph.tensors().items():
        if value_tensor.dtype != np.float16:
            continue
        if value_tensor.__class__ != gs.Constant:
            if verbose:
                print(f"Converting float16 tensor to float 32 for tensor {name}")
            value_tensor.dtype = np.float32
            continue
        original_output_tensor_name = name
        if original_output_tensor_name in already_processed:
            continue
        already_processed.add(original_output_tensor_name)
        if verbose:
            print(f"Processing tensor {name}")
        convert_f16_to_f32_tensor(
            name, value_tensor, original_output_tensor_name, graph, verbose
        )

    return already_processed


def fix_untyped_casts(output_filename, verbose=False, ir_version=None):
    """For some reason the integer activations quantization tool generates casts with no type attribute, so patch those up"""
    if verbose:
        print(f"Fixing untyped casts in {output_filename}")
    
    model = onnx.load(output_filename)
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.op == "Cast":
            if "to" not in node.attrs:
                if verbose:
                    print(f"Fixing untyped cast {node.name}")
                node.attrs["to"] = 1
    new_model = gs.export_onnx(graph)
    new_model.graph.value_info.clear()
    if ir_version is not None:
        new_model.ir_version = ir_version
    onnx.save(new_model, output_filename)


if __name__ == "__main__":
    """Command line utility to quantize ONNX models."""
    import argparse
    import glob
    import os
    import sys

    def get_list_arg(arg):
        if arg is None:
            return None
        return arg.split(",")

    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Quantization utility for ONNX models",
    )
    parser.add_argument(
        "--method", # Deprecated name, now using --action
        "-m",
        "--action",
        "-a",
        help="How to quantize the models",
        default="integer_weights",
        choices=[
            "integer_weights",
            "float_weights",
            "integer_activations",
            "f16_to_f32",
            "to-text-proto"
        ],
    )
    parser.add_argument(
        "--float_levels",
        "-l",
        help="Number of levels to use for float quantization.",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Folder to write the quantized models to. If not specified, uses the same folder as the input models.",
        default=None,
    )
    parser.add_argument(
        "--output_suffix",
        "-s",
        help="Suffix to add to the output model filenames.",
        default=None,
    )
    parser.add_argument(
        "--op_types_to_quantize",
        "-q",
        help="Comma-separated list of op types to quantize (default is all supported).",
        default=None,
    )
    parser.add_argument(
        "--nodes_to_quantize",
        "-t",
        help="Comma-separated list of node names to quantize (default is all).",
        default=None,
    )
    parser.add_argument(
        "--nodes_to_exclude",
        "-n",
        help="Comma-separated list of node names not to quantize (default is none).",
        default=None,
    )
    parser.add_argument(
        "--info",
        "-i",
        help="Whether to print information about the weights in the model.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="Log detailed information about the weight processing.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save-protos",
        "-p",
        help="Write out the input and output ONNX files as text protobufs.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--external-data",
        "-e",
        help="Use external data for the output ONNX files.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ir-version",
        "-r",
        help="The IR version to use for the output ONNX files.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--default-tensor-type",
        "-d",
        help="The default tensor type to use for integer activation quantization (1 is float32).",
        default=None,
        type=int,
    )
    parser.add_argument("globs", nargs="*")
    args = parser.parse_args()
    if len(args.globs) == 0:
        args.globs = ["*.onnx"]
    if args.output_suffix is None:
        if args.method == "integer_activations":
            args.output_suffix = "_quantized_activations.onnx"
        elif args.method == "f16_to_f32":
            args.output_suffix = "_f16_to_f32.onnx"
        else:
            args.output_suffix = "_quantized_weights.onnx"

    if args.output_dir is not None and not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    op_types_to_quantize = get_list_arg(args.op_types_to_quantize)
    nodes_to_quantize = get_list_arg(args.nodes_to_quantize)
    nodes_to_exclude = get_list_arg(args.nodes_to_exclude)

    for input_glob in args.globs:
        if os.path.isdir(input_glob):
            input_glob = os.path.join(input_glob, "*.onnx")
        input_filenames = list(glob.glob(input_glob))
        if len(input_filenames) == 0:
            print(f"No files found matching '{input_glob}'.")
            sys.exit(1)

        for input_filename in input_filenames:
            if args.info:
                print_weight_info(input_filename)
                continue
            if args.output_suffix != ".onnx" and input_filename.endswith(
                args.output_suffix
            ):
                print(f"Skipping '{input_filename}' as it is already quantized.")
                continue
            if args.verbose:
                print(f"Processing '{input_filename}'")
            input_base = os.path.basename(input_filename)
            input_dir = os.path.dirname(input_filename)
            output_base = os.path.splitext(input_base)[0] + args.output_suffix
            if args.output_dir is None:
                output_filename = os.path.join(input_dir, output_base)
            else:
                output_filename = os.path.join(args.output_dir, output_base)
            if args.verbose:
                print(f"Writing to '{output_filename}'")
            if output_filename == input_filename:
                print(
                    f"Skipping '{input_filename}' as the output filename is the same and it would be overwritten."
                )
                continue
            if args.verbose:
                input_file_length = os.path.getsize(input_filename)
            if args.method == "float_weights" or args.method == "integer_weights":
                original_model = onnx.load(input_filename)
                float_quantization = args.method == "float_weights"
                new_model = quantize_weights(
                    original_model,
                    float_quantization=float_quantization,
                    float_levels=args.float_levels,
                    verbose=args.verbose,
                    ir_version=args.ir_version,
                )
                if args.verbose:
                    print(
                        f"Saving model converted from {input_filename} to {output_filename}"
                    )
                onnx.save(
                    new_model, output_filename, save_as_external_data=args.external_data
                )
            elif args.method == "integer_activations":
                if args.verbose:
                    print(
                        f"quantize_dynamic('{input_filename}', '{output_filename}', weight_type=QuantType.QUInt8, op_types_to_quantize={op_types_to_quantize}, nodes_to_quantize={nodes_to_quantize}, nodes_to_exclude={nodes_to_exclude}, extra_options={{'EnableSubgraph': True}})"
                    )
                quantize_dynamic(
                    input_filename,
                    output_filename,
                    weight_type=QuantType.QUInt8,
                    op_types_to_quantize=op_types_to_quantize,
                    nodes_to_quantize=nodes_to_quantize,
                    nodes_to_exclude=nodes_to_exclude,
                    extra_options={
                        "EnableSubgraph": True,
                        "DefaultTensorType": args.default_tensor_type,
                    },
                )
                fix_untyped_casts(output_filename, args.verbose, args.ir_version)
            elif args.method == "f16_to_f32":
                original_model = onnx.load(input_filename)
                new_model = convert_f16_to_f32(
                    input_model=original_model,
                    ir_version=args.ir_version,
                    verbose=args.verbose,
                    nodes_to_exclude=nodes_to_exclude,
                )
                new_model.graph.value_info.clear()
                if args.verbose:
                    print(
                        f"Saving model converted from {input_filename} to {output_filename}"
                    )
                onnx.save(
                    new_model, output_filename, save_as_external_data=args.external_data
                )
            elif args.method == "to-text-proto":
                original_model = onnx.load(input_filename)
                tmp_filename = input_filename + ".tmp.txt"
                output_filename = input_filename + ".txt"
                with open(tmp_filename, "w") as f:
                    f.write(str(original_model))
                with open(output_filename, "w") as f:
                    with open(tmp_filename, "r") as f2:
                        for line in f2:
                            if "raw_data" in line:
                                continue
                            f.write(line)
                os.remove(tmp_filename)
            else:
                print(f"Unknown quantization method: {args.method}")
                sys.exit(1)
            if args.verbose:
                output_file_length = os.path.getsize(output_filename)
                print(
                    f"Original file size: {input_file_length:,} bytes, quantized file size: {output_file_length:,} bytes"
                )
