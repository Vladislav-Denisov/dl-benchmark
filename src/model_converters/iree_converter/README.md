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
- `-is / --input_shape` is an input shape in the format BxWxHxC, where B is a batch size, W is an input tensor width, H is an input tensor height, C is an input tensor number of channels. Required for PyTorch models.
- `--onnx_opset_version` is the ONNX opset version for ONNX models. Default: `18`.
- `-o / --output_mlir` is path to save the MLIR file. Required.

### Parameter combinations
#### For ONNX models:
- Required: `--source_framework onnx`, `--model <path/to/model.onnx>`, `--output_mlir <output_path>`
- Optional: `--onnx_opset_version` (default: 18)
#### For PyTorch models:
Two loading methods are supported (mutually exclusive):
1. From file:
- Required: `--source_framework pytorch`, `--model <path/to/model.pt>`, `--input_shape B W H C`, `--output_mlir <output_path>`
- Optional: `--weights <path/to/weights.pth>`
1. From module:
- Required: `--source_framework pytorch`, `--model_name <model_name>`, `--torch_module <module>`, `--input_shape B W H C`, `--output_mlir <output_path>`
- Optional: `--weights <path/to/weights.pth>`

### Examples of usage
ONNX model conversion:
```sh
python3 iree_converter.py -f onnx -m efficientnet-b0.onnx \
                         --onnx_opset_version 18 \
                         -o ./output/efficientnet-b0.mlir
```

PyTorch model from file:
```sh
python3 iree_converter.py -f pytorch -m resnet50.pt \
                         -is 1 224 224 3 \
                         -o ./output/resnet50.mlir
```

PyTorch model from torchvision with pretrained weights:
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
- `-m / --mlir` - Path to an .mlir file with a model. Required.
- `-tb / --target_backend` - Target backend for compilation. Required. Examples: `llvm-cpu`, `cuda`, `vulkan`, `vmvx`.
- `--opt_level` - The optimization level of the compilation. Choices: `0`, `1`, `2`, `3`. Default: `2`.
- `-o / --output_file` - Path to save the compiled model. Required.
- `--extra_args` - Extra arguments for compilation. Optional.

### Supported target backends
- `llvm-cpu` - CPU execution using LLVM
- `cuda` - NVIDIA GPU execution using CUDA
- `vulkan` - GPU execution using Vulkan API
- `vmvx` - Portable VM bytecode execution
- `metal` - Apple GPU execution using Metal
- `rocm` - AMD GPU execution using ROCm

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
