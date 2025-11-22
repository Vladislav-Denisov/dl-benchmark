import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('model_converters',
                                                                 'iree_converter',
                                                                 'iree_auxiliary')))
from compiler import IREECompiler  # noqa: E402
from converter import IREEConverter  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('utils')))
from logger_conf import configure_logger  # noqa: E402

log = configure_logger()

try:
    import iree.runtime as ireert  # noqa: E402
except ImportError as e:
    log.error(f'IREE import error: {e}')
    sys.exit(1)


def _validate_iree_model_args(args):
    if not args.model:
        raise ValueError('Model path (-m/--model) is required')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'The file not found: {args.model}')

    file_type = args.model.split('.')[-1].lower()
    supported_extensions = ['mlir', 'vmfb']
    if file_type not in supported_extensions:
        raise ValueError(f'Model must be an {supported_extensions} file')
    if file_type == 'mlir' and not args.target_backend:
        raise ValueError('target_backend is required when using .mlir model')


def _validate_onnx_args(args):
    if not args.model:
        raise ValueError('Model path (-m/--model) is required for ONNX framework')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model file not found: {args.model}')

    file_type = args.model.split('.')[-1]
    if file_type == 'onnx':
        if not args.onnx_opset_version:
            raise ValueError('onnx_opset_version is required for ONNX framework')
    else:
        _validate_iree_model_args(args)


def _validate_pytorch_args(args):
    has_model_path = args.model is not None and args.model != ''
    has_module_model = (args.torch_module is not None and args.torch_module != ''
                        and args.model_name is not None and args.model_name != '')

    if not has_model_path and not has_module_model:
        raise ValueError(
            'For PyTorch conversion, you must specify either model_path, '
            'or torch_module and model_name',
        )

    if has_model_path and has_module_model:
        raise ValueError(
            'Provided incompatible parameters for PyTorch conversion (model_path and torch_module+model_name). '
            'Please choose only one method.',
        )

    if has_model_path:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f'Model file not found: {args.model}')

        file_type = args.model.split('.')[-1]
        if file_type != 'pt':
            _validate_iree_model_args(args)
    else:
        if not args.target_backend:
            raise ValueError('target_backend is required when using conversion from torch module')

    if args.model_weights and args.model_weights != '' and not os.path.exists(args.model_weights):
        raise FileNotFoundError(f'Model weights not found: {args.model_weights}')


def validate_cli_args(args):
    if args.source_framework == 'onnx':
        _validate_onnx_args(args)
    elif args.source_framework == 'pytorch':
        _validate_pytorch_args(args)
    else:
        _validate_iree_model_args(args)


def _convert_model_to_mlir(model_path, model_weights, torch_module, model_name, onnx_opset_version,
                           source_framework, input_shape, output_mlir):
    dictionary = {
        'source_framework': source_framework,
        'model_name': model_name,
        'model_path': model_path,
        'model_weights': model_weights,
        'torch_module': torch_module,
        'onnx_opset_version': onnx_opset_version,
        'input_shape': input_shape,
        'output_mlir': output_mlir,
    }
    converter = IREEConverter.get_converter(dictionary)
    converter.convert_to_mlir()
    return


def _compile_mlir(mlir_path, target_backend, opt_level, extra_compile_args):
    try:
        log.info('Starting model compilation')
        return IREECompiler.compile_model(mlir_path, target_backend, opt_level, extra_compile_args)
    except Exception as e:
        log.error(f'Failed to compile MLIR: {e}')
        raise


def _load_model_buffer(model_path, target_backend, opt_level, extra_compile_args):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    file_type = model_path.split('.')[-1]

    if file_type == 'mlir':
        if target_backend is None:
            raise ValueError('target_backend is required for MLIR compilation')
        vmfb_buffer = _compile_mlir(model_path, target_backend, opt_level, extra_compile_args)
    elif file_type == 'vmfb':
        with open(model_path, 'rb') as f:
            vmfb_buffer = f.read()
    else:
        raise ValueError(f'The file type {file_type} is not supported. Supported types: .mlir, .vmfb')

    log.info(f'Successfully loaded model buffer from {model_path}')
    return vmfb_buffer


def _create_iree_context_from_buffer(vmfb_buffer):
    try:
        config = ireert.Config('local-task')
        vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, vmfb_buffer)
        context = ireert.SystemContext(config=config)
        context.add_vm_module(vm_module)

        log.info('Successfully created IREE context from buffer')
        return context

    except Exception as e:
        log.error(f'Failed to create IREE context: {e}')
        raise


def load_model(model_path, model_weights, torch_module, model_name, onnx_opset_version,
               source_framework, input_shape, target_backend, opt_level, extra_compile_args):
    is_tmp_mlir = False
    if model_path is None or model_path.split('.')[-1] not in ['vmfb', 'mlir']:
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False, suffix='.mlir') as temp:
            output_mlir = temp.name
            _convert_model_to_mlir(model_path,
                                   model_weights,
                                   torch_module,
                                   model_name,
                                   onnx_opset_version,
                                   source_framework,
                                   input_shape,
                                   output_mlir)
            model_path = output_mlir
            is_tmp_mlir = True

    vmfb_buffer = _load_model_buffer(
        model_path,
        target_backend=target_backend,
        opt_level=opt_level,
        extra_compile_args=extra_compile_args,
    )

    if is_tmp_mlir:
        os.remove(model_path)

    return _create_iree_context_from_buffer(vmfb_buffer)


def prepare_output(result, task):
    if task == 'feedforward':
        return {}
    elif task == 'classification':
        if hasattr(result, 'to_host'):
            result = result.to_host()

        # Extract tensor from dict if needed
        if isinstance(result, dict):
            result_key = next(iter(result))
            logits = result[result_key]
            output_key = result_key
        else:
            logits = np.array(result)
            output_key = 'output'

        # Ensure correct shape (batch_size, num_classes)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        elif logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)

        # Apply softmax
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return {output_key: probabilities}
    else:
        raise ValueError(f'Unsupported task {task}')


def create_dict_for_transformer(args):
    return {
        'channel_swap': args.channel_swap,
        'mean': args.mean,
        'std': args.std,
        'norm': args.norm,
        'layout': args.layout,
        'input_shape': args.input_shape,
        'batch_size': args.batch_size,
    }
