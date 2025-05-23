import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

from executorch.runtime import Runtime, Verification
import numpy as np

from io_adapter import IOAdapter
from io_model_wrapper import OpenCVIOModelWrapper
from transformer import Transformer

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('utils')))
from logger_conf import configure_logger, exception_hook  # noqa: E402

log = configure_logger()


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bch', '--benchmark_app',
                        help='Path to executorch cpp benchmark',
                        required=True,
                        type=str,
                        dest='benchmark_path')
    parser.add_argument('-m', '--model',
                        help='Path to a .pte file with a trained model.',
                        required=True,
                        type=str,
                        dest='model_path')
    parser.add_argument('-i', '--input',
                        help='Path to data',
                        required=True,
                        type=str,
                        nargs='+',
                        dest='input')
    parser.add_argument('-d', '--device',
                        help='Specify the target device to infer on (CPU by default)',
                        default='CPU',
                        type=str,
                        dest='device')
    parser.add_argument('-b', '--batch',
                        help='Batch size',
                        required=False,
                        default=1,
                        type=int,
                        dest='batch_size')
    parser.add_argument('--input_type',
                        help='Parameter input type in format [value]',
                        default='[FP32]',
                        type=str,
                        dest='input_type')
    parser.add_argument('--layout',
                        help='Parameter input layout in format [value]',
                        default=None,
                        type=str,
                        dest='layout')
    parser.add_argument('-is', '--input_shapes',
                        help='Input tensor shapes',
                        default=None,
                        type=str,
                        dest='shape')
    parser.add_argument('-in', '--input_name',
                        help='Input layer name',
                        default='_input',
                        type=str,
                        dest='input_name')
    parser.add_argument('--mean',
                        help='Mean values in <[R,G,B]>',
                        required=False,
                        default='',
                        type=str,
                        dest='mean')
    parser.add_argument('--scale',
                        help='Scale values in <[R,G,B]>',
                        required=False,
                        default='',
                        type=str,
                        dest='input_scale')
    parser.add_argument('--swap_channels',
                        help='Parameter channel swap',
                        required=False,
                        default=False,
                        type=bool,
                        dest='swap_channels')
    parser.add_argument('--background',
                        help='Path to background image',
                        type=str,
                        dest='background')
    parser.add_argument('--time',
                        help='Optional. Time in seconds to execute topology.',
                        required=False,
                        type=int,
                        default=0,
                        dest='time_limit')
    parser.add_argument('-ni', '--number_iter',
                        help='Number of inference iterations',
                        default=1,
                        type=int,
                        dest='number_iter')
    parser.add_argument('-t', '--task',
                        help='Output processing method. Default: without postprocess',
                        choices=['classification'],
                        default='feedforward',
                        type=str,
                        dest='task')
    parser.add_argument('--use_bin_input',
                        help='Use binary input',
                        required=False,
                        default=False,
                        type=bool,
                        dest='use_bin_input')
    parser.add_argument('--output_json_path',
                        help='Path to save raw output of cpp_dl_benchmark',
                        type=Path,
                        dest='output_json_path')
    parser.add_argument('--output_path',
                        help='Path to save processed output',
                        type=Path,
                        dest='output_path')
    parser.add_argument('--only_process_output',
                        help='Run without model execution',
                        required=False,
                        default=False,
                        type=bool,
                        dest='only_process_output')
    parser.add_argument('-nt', '--number_top',
                        help='Number of top results',
                        default=5,
                        type=int,
                        dest='number_top')
    parser.add_argument('--labels',
                        help='Labels mapping file',
                        default=None,
                        type=str,
                        dest='labels')

    args = parser.parse_args()

    return args


class ExecuTorchProcess():
    def __init__(self):
        self._command_line = ''

    def _add_argument(self, name_of_arg, value_of_arg):
        self._command_line += f' {name_of_arg} {value_of_arg}'

    def _add_option(self, name_of_arg):
        self._command_line += f' {name_of_arg}'

    def create_command_line(self, dict_of_args):
        for name, arg in dict_of_args.items():
            if name == '-bch':
                self._add_option(arg)
            elif arg != '':
                self._add_argument(name, arg)
        self._add_argument('--nthreads', 1)
        self._add_option('--dump_output')

    def execute(self):
        log.info(f'Command line: {self._command_line}')
        proc = subprocess.run(self._command_line, shell=True)
        if proc.returncode != 0:
            log.error(traceback.format_exc())
            sys.exit(1)

    def process_benchmark_output(self, output_filename):
        result = {}
        with open(output_filename, 'r') as file:
            for output in json.load(file)[0]:
                layer_name = output['output_name']
                shape = output['shape']
                data = output['data']
                result[layer_name] = np.reshape(data, shape)
        return result


def get_output_json_path(args):
    if args.output_json_path is None:
        return Path(__file__).parent / '_validation' / 'json_output' / 'output.json'
    return Path(args.output_json_path)


def create_dict_for_wrapper(args):
    dictionary = {
        'input_layer_name': args.input_name,
        'input_layer_shape': [args.batch_size, *args.shape],
    }
    return dictionary


def create_dict_from_args_for_process(args, io):
    args_dict = {'-bch': args.benchmark_path,
                 '--niter': args.number_iter,
                 '-t': args.time_limit,
                 '-m': args.model_path,
                 '-d': args.device,
                 '-b': args.batch_size,
                 '--layout': args.layout,
                 '--shape': args.shape,
                 '--dtype': args.input_type,
                 '--output_path': args.output_json_path}
    args_dict['--mean'] = args.mean
    args_dict['--scale'] = args.input_scale
    args_dict['--channel_swap'] = '' if not args.swap_channels else True
    args_dict['-i'] = args.input[0]
    return args_dict


def load_network(model):
    ex_program = Runtime.get()
    program = ex_program.load_program(
        Path(model),
        verification=Verification.Minimal,
    )
    net = program.load_method('forward')
    return net


def main():
    sys.excepthook = exception_hook

    args = cli_argument_parser()
    try:
        log.info(f'Loading network files:\n\t {args.model_path}')
        net = load_network(args.model_path)

        model_wrapper = OpenCVIOModelWrapper(create_dict_for_wrapper(args))

        args.input_names = model_wrapper.get_input_layer_names(net)
        args.output_json_path = get_output_json_path(args)

        layer_name = model_wrapper.get_input_layer_names(net)
        input_shape = model_wrapper.get_input_layer_shape(net, layer_name)
        log.info(f'Shape for input layer {layer_name}: {input_shape}')

        data_transformer = Transformer()
        io = IOAdapter.get_io_adapter(args, model_wrapper, data_transformer)

        log.info('Initializing ExecuTorch process')
        process = ExecuTorchProcess()
        process.create_command_line(create_dict_from_args_for_process(args, io))

        if not args.only_process_output:
            log.info('ExecuTorch benchmark process:\n')
            process.execute()

        log.info('Process benchmark output:\n')
        result = process.process_benchmark_output(args.output_json_path)

        log.info('Process output using io_adapter:\n')
        io.process_output(result, log)

    except Exception:
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main() or 0)
