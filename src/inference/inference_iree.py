import argparse
import sys
import traceback
from pathlib import Path

import postprocessing_data as pp
from inference_tools.loop_tools import loop_inference, get_exec_time
from io_adapter import IOAdapter
from io_model_wrapper import IREEModelWrapper
from reporter.report_writer import ReportWriter
from transformer import IREETransformer

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('utils')))
from logger_conf import configure_logger  # noqa: E402

log = configure_logger()

try:
    import iree.runtime as ireert  # noqa: E402
except ImportError as e:
    log.error(f"IREE import error: {e}")
    sys.exit(1)


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    
    parser.add_argument('-m', '--model',
                        help='Path to .vmfb file with compiled model.',
                        required=True,
                        type=str,
                        dest='model')
    parser.add_argument('-fn', '--function_name',
                        help='IREE module function name to execute.',
                        required=True,
                        type=str,
                        dest='function_name')
    parser.add_argument('-i', '--input',
                        help='Path to data.',
                        required=True,
                        type=str,
                        nargs='+',
                        dest='input')
    parser.add_argument('-is', '--input_shape',
                        help='Input shape BxHxWxC, B is a batch size,'
                             'H is an input tensor height,'
                             'W is an input tensor width,'
                             'C is an input tensor number of channels.',
                        required=True,
                        type=int,
                        nargs=4,
                        dest='input_shape')
    parser.add_argument('-b', '--batch_size',
                        help='Size of the processed pack.'
                             'Should be the same as B in input_shape argument.',
                        default=1,
                        type=int,
                        dest='batch_size')
    parser.add_argument('-l', '--labels',
                        help='Labels mapping file.',
                        default=None,
                        type=str,
                        dest='labels')
    parser.add_argument('-nt', '--number_top',
                        help='Number of top results.',
                        default=5,
                        type=int,
                        dest='number_top')
    parser.add_argument('-t', '--task',
                        help='Task type. Default: feedforward.',
                        choices=['feedforward', 'classification'],
                        default='feedforward',
                        type=str,
                        dest='task')
    parser.add_argument('-ni', '--number_iter',
                        help='Number of inference iterations.',
                        default=1,
                        type=int,
                        dest='number_iter')
    parser.add_argument('--raw_output',
                        help='Raw output without logs.',
                        default=False,
                        type=bool,
                        dest='raw_output')
    parser.add_argument('--time',
                        required=False,
                        default=0,
                        type=int,
                        dest='time',
                        help='Optional. Maximum test duration. 0 if no restrictions.')
    parser.add_argument('--report_path',
                        type=Path,
                        default=Path(__file__).parent / 'iree_inference_report.json',
                        dest='report_path')
    parser.add_argument('--layout',
                        help='Input layout.',
                        default='NHWC',
                        choices=['NHWC', 'NCHW'],
                        type=str,
                        dest='layout')
    parser.add_argument('--norm',
                        help='Flag to normalize input images.',
                        action='store_true',
                        dest='norm')
    parser.add_argument('--mean',
                        help='Mean values.',
                        default=[0, 0, 0],
                        type=float,
                        nargs=3,
                        dest='mean')
    parser.add_argument('--std',
                        help='Standard deviation values.',
                        default=[1., 1., 1.],
                        type=float,
                        nargs=3,
                        dest='std')
    parser.add_argument('--channel_swap',
                        help='Parameter of channel swap.',
                        default=[2, 1, 0],
                        type=int,
                        nargs=3,
                        dest='channel_swap')
    parser.add_argument('-d', '--device',
                        help='Specify the target device to infer (CPU by default)',
                        default='CPU',
                        type=str,
                        dest='device')

    return parser.parse_args()


def load_iree_model(model_path):
    try:
        config = ireert.Config('local-task')

        with open(model_path, 'rb') as f:
            vmfb_buffer = f.read()

        vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, vmfb_buffer)
        context = ireert.SystemContext(config=config)
        context.add_vm_module(vm_module)

        log.info(f"Successfully loaded IREE model")
        return context

    except Exception as e:
        log.error(f"Failed to load IREE model: {e}")
        raise


def get_inference_function(model_context, function_name):
    try:
        main_module = model_context.modules.module
        inference_func = main_module[function_name]
        log.info(f"Using function '{function_name}' for inference")
        return inference_func

    except Exception as e:
        log.error(f"Failed to get inference function: {e}")
        raise


def inference_iree(inference_func, number_iter, get_slice, test_duration):
    result = None
    time_infer = []

    if number_iter == 1:
        slice_input = get_slice()
        result, exec_time = infer_slice(inference_func, slice_input)
        time_infer.append(exec_time)
    else:
        time_infer = loop_inference(number_iter, test_duration)(
            inference_iteration
        )(inference_func, get_slice)['time_infer']
    
    log.info('Inference completed')
    return result, time_infer


def inference_iteration(inference_func, get_slice):
    slice_input = get_slice()
    _, exec_time = infer_slice(inference_func, slice_input)
    return exec_time


@get_exec_time()
def infer_slice(inference_func, slice_input):
    config = ireert.Config('local-task')
    device = config.device

    input_buffers = list()
    for input_ in slice_input:
        input_buffers.append(ireert.asdevicearray(device, input_))
    
    result = inference_func(*input_buffers)

    if hasattr(result, 'to_host'):
        result = result.to_host()

    return result


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
        'channel_swap': getattr(args, 'channel_swap'),
        'mean': getattr(args, 'mean'),
        'std': getattr(args, 'std'),
        'norm': getattr(args, 'norm'),
        'layout': getattr(args, 'layout'),
        'input_shape': getattr(args, 'input_shape'),
        'batch_size': getattr(args, 'batch_size'),
    }


def main():
    args = cli_argument_parser()
    
    try:
        model_wrapper = IREEModelWrapper(args)
        data_transformer = IREETransformer(create_dict_for_transformer(args))
        io = IOAdapter.get_io_adapter(args, model_wrapper, data_transformer)

        report_writer = ReportWriter()
        report_writer.update_framework_info(name='IREE')
        report_writer.update_configuration_setup(
            batch_size=args.batch_size,
            iterations_num=args.number_iter,
            target_device=args.device
        )

        model_context = load_iree_model(args.model)
        inference_func = get_inference_function(model_context, args.function_name)

        log.info(f'Preparing input data: {args.input}')
        io.prepare_input(model_context, args.input)

        log.info(f'Starting inference ({args.number_iter} iterations) on {args.device}')
        result, inference_time = inference_iree(
            inference_func,
            args.number_iter,
            io.get_slice_input_iree,
            args.time
        )

        log.info('Computing performance metrics')
        inference_result = pp.calculate_performance_metrics_sync_mode(
            args.batch_size, 
            inference_time
        )
    
        report_writer.update_execution_results(**inference_result)
        report_writer.write_report(args.report_path)

        if not args.raw_output:
            if args.number_iter == 1:
                try:
                    log.info('Converting output tensor to print results')
                    result = prepare_output(result, args.task)
                    log.info('Inference results')
                    io.process_output(result, log)
                except Exception as ex:
                    log.warning(f'Error when printing inference results: {str(ex)}')

        log.info(f'Performance results: {inference_result}')

    except Exception:
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main() or 0)
