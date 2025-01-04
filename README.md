# ONNX Shrink Ray

Shrinks the size of ONNX files by quantizing large float constants into eight bit equivalents, while leaving all calculations in floating point.

- [Installation](#installation)
- [Usage](#usage)
  - [To reduce the size of a single file](#to-reduce-the-size-of-a-single-file)
  - [To reduce the compressed size of a file](#to-reduce-the-compressed-size-of-a-file)
  - [To print information about the weights in a file](#to-print-information-about-the-weights-in-a-file)
- [What Shrink Ray does](#what-shrink-ray-does)
- [Results](#results)
  - [Moonshine Tiny](#moonshine-tiny)
  - [Moonshine Base](#moonshine-base)
  - [Notes](#notes)
 - [Other Models](#other-models)
  
## Installation

The easiest way to get started is to install this package in Python using pip:

```bash
pip install onnx_shrink_ray
```

You can also download this repository and run the `shrink.py` script directly.

## Usage

### To reduce the size of a single file

```bash
python -m onnx_shrink_ray.shrink myfile.onnx
```

This will convert all of the weights in the ONNX file from 32-bit floating point to 8-bit integers, followed by a `DequantizeLinear` operation to linearly scale those into approximations of the original values for later calculations. The resulting ONNX file is typically less than 30% of the input's size.

### To reduce the compressed size of a file

```bash
python -m onnx_shrink_ray.shrink --method "float_weights" --float_levels 256 myfile.onnx
```

A lot of downloads and app bundles are automatically compressed using a standard like `gzip` or `brotli`. Neural network weights often don't compress well when they're stored as floating point numbers, since there is very little repetition in the values, they're usually all slightly different from one another. If we know our model will be compressed for delivery, we can reduce the actual download size by making the weight values (which normally make up the majority of the file) easier to compress. 

This tool does this by rounding all the float values in a weight array to the nearest in a limited number of quantized steps, but then storing the results back into a 32-bit floating point tensor. This means the uncompressed size on disk remains the same, but the compressed version is often several times smaller. This is because there's now only a limited number of values in each weight tensor, so there's a lot more repetition in the byte stream for the compression algorithm to take advantage of.

By default, each weight tensor is quantized to 256 levels, but since the results are stored as floating point values, you can modify this to trade off compressed file size for accuracy. For example, increasing the `--float_levels` argument to 1,000 can improve accuracy at the cost of a larger compressed file, whereas 100 would shrink the size, but could negatively impact quality.

### To print information about the weights in a file

```bash
python -m onnx_shrink_ray.shrink --info myfile.onnx
```

This will analyze the file, and output information about the weight arrays stored in it, including their shape, type, and size in bytes. It will also show how much of the file size is weights, and how much is from other information. Ideally, the weights should be the majority of the file size. Here is some example output:

```bash
Model: decoder_model_merged.onnx
Initializer: onnx::MatMul_2282_merged_0_quantized: [288, 288] - 82,944 elements, uint8, 82,944 bytes
...
Initializer: onnx::MatMul_2444_merged_0_quantized: [1152, 288] - 331,776 elements, uint8, 331,776 bytes
Initializer: model.decoder.embed_tokens.weight_merged_0_quantized: [32768, 288] - 9,437,184 elements, int8, 9,437,184 bytes 
Total nodes: 0
Total initializers: 61
Total bytes from weights: 19,475,173 bytes, 9,819,391 bytes from other data
-------------------------------------------
```

## What Shrink Ray does

Standard ONNX quantization is focused on converting all calculations to eight bit, which can reduce latency dramatically on some platforms. This approach can also cause accuracy problems however, and often requires some manual work to achieve the best results.

Sometimes though, the biggest problem is not speeding up the execution of a network, but reducing the size of the model data. This can be the case when a model has to be downloaded, where the size determines the loading time before it can be used, or when it's part of a mobile app bundle or other edge device with limited storage space.

The standard ONNX quantization does offer some file size benefits, but the potential impact on accuracy means it can take time and effort to achieve these savings. As an alternative, this module implements "weight-only quantization", where all calculations and activation layers are left in their initial precision, and only the weights are stored in a lower-fidelity format.

This approach has the advantage that it is much less likely to significantly impact accuracy, and so can usually be applied quickly, with no manual tweaking or fixups required. It will not speed up latency (and some of the methods may actually slow execution by a small amount) but it can offer significant file size savings.

Though this method is designed to have a minimal impact on the accuracy of the model, there are networks that may be adversely affected. The heuristic used to identify weights simply searches for constants or initializers that are larger than 16,384 elements, with the assumption that smaller constants are more likely to be non-weight parameters, and won't contribute much to the overall size of the model on disk.

## Results

The initial reason for creating this project was to reduce the download size for the [Moonshine](https://github.com/usefulsensors/moonshine) models on the web, so I've done the most extensive testing on those networks. Here are the size and accuracy results when running against the LibreSpeech clean English-language dataset.

### Moonshine Tiny

|                              | WER    | File Size | GZIP Size | Brotli Size | Latency |
|------------------------------|--------|-----------|-----------|-------------|---------|
| Original                     | 4.51%  | 272MB	    | 251MB	    | 226MB       | 307ms   |
| Integer Weights              | 4.69%  | 69MB	    | 53MB	    | 46MB        | 466ms   |
| Float Weights (100 levels)   | 11.34% | 272MB	    | 60MB	    | 46MB        | 188ms   |
| Float Weights (256 levels)   | 4.69%  | 272MB	    | 75MB	    | 59MB        | 329ms   |
| Float Weights (1,000 levels) | 4.47%  | 272MB     | 108MB	    | 79MB        | 296ms   |
| ONNX Dynamic Quantization	   | 30.99% | 113MB	    | 95MB	    | 71MB        | 317ms   |

### Moonshine Base

|                              | WER    | File Size | GZIP Size | Brotli Size | Latency |
|------------------------------|--------|-----------|-----------|-------------|---------|
| Original                     | 3.29%  | 556MB	    | 515MB	    | 469MB       | 420ms   |
| Integer Weights              | 3.28%  | 141MB	    | 105MB	    | 92MB        | 729ms   |
| Float Weights (100 levels)   | 3.55%  | 556MB	    | 120MB	    | 94MB        | 402ms   |
| Float Weights (256 levels)   | 3.28%  | 556MB	    | 155MB	    | 121MB       | 407ms   |
| Float Weights (1,000 levels) | 3.29%  | 556MB     | 217MB		| 161MB       | 411ms   |
| ONNX Dynamic Quantization	   | 19.06% | 264MB	    | 225MB	    | 180MB       | 221ms   |

### Notes

The compressed file sizes were calculated by checking the archive size after running `tar --use-compress-program="<brotli|gzip> --best" -cvf archive.tbz <folder of model files>`. The `--best` flag is used here to ensure the compression is as effective as possible by running multiple passes.

Latency values were calculated by running a ten second audio clip through each model on a Microsoft Surface Pro with an x86 CPU, using the `moonshine_onnx.benchmark()` function included in the library.

ONNX dynamic quantization results are included for reference. These are models produced by the [`onnxruntime.quantization.quantize_dynamic()`](https://iot-robotics.github.io/ONNXRuntime/docs/performance/quantization.html#quantization-api) function with default arguments. For convenience you can invoke this through the `--method "integer_activations"` option.

Some interesting patterns are visible:

 - The float weight quantization has no effect on the uncompressed file size, but dramatically decreases the compressed file size, as expected. It also has makes no statistically significant difference to the latency.

 - The integer weight quantization is a lot slower than float weights. This is a bit surprising, since the only difference is a DequantizeLinear operation for each weight constant, but my best guess is that the op hasn't been optimized, on this platform at least.

 - ONNX quantization produces models that are fast, but much less accurate. In my experience this is a common outcome, and can be fixed with some investigation into exactly where the accuracy loss is occuring, but it tends to be a time-consuming process, hence my desire for something easier when file size is the biggest obstacle.

 - ONNX quantization doesn't shrink the raw files as much as I'd expect. If the weights were being stored as 8-bit integers, I'd expect the file size to be the same as the `integer_weights` version, but they're about twice as large. I wonder if the weights are actually stored as 16-bit in this case, or if there's somehow an extra copy?

 - Different models can tolerate different levels of float quantization. The base model only loses a fraction of a percent at 100 levels, whereas the tiny model loses several points.

 - Brotli does a better job at compressing these files than gzip, though the compression process takes significantly longer in my experience. Since brotli is now widely supported by browsers, it seems like the best method to use overall.

 - Apart from the integer weights, most of the float weights versions have similar latencies to the original model. This is expected, since the overall network architecture isn't changed, just the values stored in constants. The only exception is the tiny float weights with 100 levels, which is unexpectedly fast. I don't have a good explanation for this yet, it will require deeper profiling.

 ## Other Models

 I haven't done widespread testing with other models to see what the quality, size, and performance impact is. I'll be maintaining this repository on a best effort basis, so though there are no guarantees on fixes, please [file an issue](https://github.com/usefulsensors/onnx_shrink_ray/issues) if you hit problems with your own models and I'll take a look.

 Pete Warden, pete@usefulsensors.com