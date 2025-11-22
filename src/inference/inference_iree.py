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
from iree_auxiliary import (load_model, create_dict_for_transformer, prepare_output, validate_cli_args)


sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('utils')))
from logger_conf import configure_logger  # noqa: E402

log = configure_logger()

try:
    import iree.runtime as ireert  # noqa: E402
except ImportError as e:
    log.error(f'IREE import error: {e}')
    sys.exit(1)


def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--source_framework',
                        help='Source model framework (required for automatic conversion to MLIR)',
                        type=str,
                        choices=['onnx', 'pytorch'],
                        dest='source_framework')
    parser.add_argument('-m', '--model',
                        help='Path to source framework model (.onnx, .pt),'
                             'to file with compiled model (.vmfb)'
                             'or MLIR (.mlir).',
                        type=str,
                        dest='model')
    parser.add_argument('-w', '--weights',
                        help='Path to an .pth file with a trained weights.'
                             'Availiable when source_framework=pytorch ',
                        type=str,
                        dest='model_weights')
    parser.add_argument('-tm', '--torch_module',
                        help='Torch module with model architecture.'
                             'Availiable when source_framework=pytorch',
                        type=str,
                        dest='torch_module')
    parser.add_argument('-mn', '--model_name',
                        help='Model name.',
                        type=str,
                        dest='model_name')
    parser.add_argument('--onnx_opset_version',
                        help='Path to an .onnx with a trained model.'
                             'Availiable when source_framework=onnx',
                        type=int,
                        dest='onnx_opset_version')
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
                        help='Input shape BxCxHxW, B is a batch size,'
                             'C is an input tensor number of channels,'
                             'H is an input tensor height,'
                             'W is an input tensor width.',
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
                        default='NCHW',
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
    parser.add_argument('-tb', '--target_backend',
                        help='Target backend, for example `llvm-cpu` for CPU.',
                        default='llvm-cpu',
                        type=str,
                        dest='target_backend')
    parser.add_argument('--opt_level',
                        help='The optimization level of the compilation.',
                        type=int,
                        choices=[0, 1, 2, 3],
                        default=2)
    parser.add_argument('--extra_compile_args',
                        help='The extra arguments for MLIR compilation.',
                        type=str,
                        nargs=argparse.REMAINDER,
                        default=[])
    args = parser.parse_args()
    validate_cli_args(args)
    return args


def get_inference_function(model_context, function_name):
    try:
        main_module = model_context.modules.module
        inference_func = main_module[function_name]
        log.info(f'Using function {function_name} for inference')
        return inference_func

    except Exception as e:
        log.error(f'Failed to get inference function: {e}')
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
            inference_iteration,
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

    input_buffers = []
    for input_ in slice_input:
        input_buffers.append(ireert.asdevicearray(device, input_))

    result = inference_func(*input_buffers)

    if hasattr(result, 'to_host'):
        result = result.to_host()

    return result


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
            target_device=args.target_backend,
        )

        log.info('Loading model')
        model_context = load_model(
            model_path=args.model,
            model_weights=args.model_weights,
            torch_module=args.torch_module,
            model_name=args.model_name,
            onnx_opset_version=args.onnx_opset_version,
            source_framework=args.source_framework,
            input_shape=args.input_shape,
            target_backend=args.target_backend,
            opt_level=args.opt_level,
            extra_compile_args=args.extra_compile_args,
        )
        inference_func = get_inference_function(model_context, args.function_name)

        log.info(f'Preparing input data: {args.input}')
        io.prepare_input(model_context, args.input)

        log.info(f'Starting inference ({args.number_iter} iterations) on {args.target_backend}')
        result, inference_time = inference_iree(
            inference_func,
            args.number_iter,
            io.get_slice_input_iree,
            args.time,
        )

        log.info('Computing performance metrics')
        inference_result = pp.calculate_performance_metrics_sync_mode(
            args.batch_size,
            inference_time,
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
