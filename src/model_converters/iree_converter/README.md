# Conversion to the IREE format
IREE converter supports conversion to the IREE MLIR format from ONNX and PyTorch formats.

IREE compiler supports compilation from `.mlir` format to the `.vmfb` format for deployment on various backends.

## IREE converter usage

Basic usage of the script:

```sh
iree_converter.py --source_framework <source_framework> \
                  --model_name <model_name> \
                  --model <model> \
                  --weights <weights> \
                  --torch_module <torch_module> \
                  --input_shape <input_shape> \
                  --onnx_opset_version <onnx_opset_version> \
                  --output_mlir <output_mlir>
```

This script converts model from `<source_framework>` to the IREE MLIR format.

### IREE converter parameters
- `-f / --source_framework` is a source framework where the model was trained. Required. Choices: `onnx`, `pytorch`.
- `-mn / --model_name` is a model name. Required for PyTorch models loaded from module.
- `-m / --model` is a path to an `.onnx` or `.pt` file with a trained model.
- `-w / --weights` is a path to an `.pth` file with trained weights for PyTorch models.
- `-tm / --torch_module` is a module with the model architecture for PyTorch models. Default: `torchvision.models`.
- `-is / --input_shape` is an input shape in the format BxHxWxC, where B is a batch size, H is an input tensor height, W is an input tensor width, C is an input tensor number of channels. Required for PyTorch models.
- `--onnx_opset_version` is an ONNX opset version for ONNX models. Default: `18`.
- `-o / --output_mlir` is a path to save the MLIR file. Required.

### Parameter combinations
#### For ONNX models:
- Required: `--source_framework onnx`, `--model <path/to/model.onnx>`, `--output_mlir <output_path>`
- Optional: `--onnx_opset_version` (default: 18; the converter validates that the value is set, so keep the default or override it explicitly)
#### For PyTorch models:
Two loading methods are supported (mutually exclusive):
1. From file:
- Required: `--source_framework pytorch`, `--model <path/to/model.pt>`, `--input_shape B H W C`, `--output_mlir <output_path>`
- Optional: `--model_name <name>` (used only for logging), `--weights <path/to/weights.pth>`
1. From module:
- Required: `--source_framework pytorch`, `--model_name <model_name>`, `--torch_module <module>`, `--input_shape B H W C`, `--output_mlir <output_path>`
- Optional: `--weights <path/to/weights.pth>`

> **Note:** `--model` and the pair `(--torch_module`, `--model_name)` are mutually exclusive. Passing both at the same time will raise a validation error (`converter.py` enforces the rule). Likewise, `--input_shape` is only validated for PyTorch conversions, so you can omit it for ONNX.

### Examples of usage
ONNX model conversion ([source of the model efficientnet-b0.onnx](https://github.com/onnx/models/blob/main/Computer_Vision/efficientnet_b0_Opset17_timm/efficientnet_b0_Opset17.onnx)):
```sh
python3 iree_converter.py -f onnx -m efficientnet-b0.onnx \
                         --onnx_opset_version 18 \
                         -o ./output/efficientnet-b0.mlir
```

PyTorch model from file (`.pt` can be created using [tutorial](https://docs.pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules)):
```sh
python3 iree_converter.py -f pytorch -m resnet50.pt \
                         -is 1 224 224 3 \
                         -o ./output/resnet50.mlir
```

PyTorch model from [torchvision](https://docs.pytorch.org/vision/main/models.html) with pretrained weights:
```sh
python3 iree_converter.py -f pytorch -mn resnet50 \
                         -tm torchvision.models \
                         -is 1 224 224 3 \
                         -o ./output/resnet50.mlir
```

PyTorch model with custom weights:
```sh
python3 iree_converter.py -f pytorch -mn resnet50 \
                         -tm torchvision.models \
                         -w ./weights/resnet50-custom.pth \
                         -is 1 224 224 3 \
                         -o ./output/resnet50-custom.mlir
```

## IREE compiler usage

Basic usage of the script:
```sh
iree_compiler.py --mlir <input.mlir> \
                 --target_backend <target_backend> \
                 --opt_level <opt_level> \
                 --output_file <output_file> \
                 [--extra_args <extra_args>]
```
This script compiles model from `.mlir` format to the deployable binary format for the specified target backend.

### IREE compiler parameters
- `-m / --mlir` is a path to an .mlir file with a model. Required.
- `-tb / --target_backend` is a target backend for compilation. Required. Examples: `llvm-cpu`, `cuda`, `vulkan`, `vmvx`.
- `--opt_level` is an optimization level of the compilation. Choices: `0`, `1`, `2`, `3`. Default: `2`.
- `-o / --output_file` is a path to save the compiled model. Required.
- `--extra_args` - is an extra arguments for compilation. Optional.

### Supported target backends
- `llvm-cpu` - CPU execution using LLVM.
- `cuda` - NVIDIA GPU execution using CUDA.
- `vulkan` - GPU execution using Vulkan API.
- `vmvx` - Portable VM bytecode execution.
- `metal` - Apple GPU execution using Metal.
- `rocm` - AMD GPU execution using ROCm.

### Examples of usage
```sh
python3 iree_compiler.py -m ./models/resnet50.mlir \
                        -tb llvm-cpu \
                        --opt_level 2 \
                        -o ./compiled/resnet50-cpu.vmfb
```
### Using extra arguments
The `--extra_args` parameter allows passing additional compilation flags:
```sh
python3 iree_compiler.py -m model.mlir -tb llvm-cpu -o output.vmfb \
                        --extra_args --iree-llvmcpu-target-triple=x86_64-linux-gnu
```
