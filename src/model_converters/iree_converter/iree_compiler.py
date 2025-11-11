import argparse
import sys
import os
import traceback
from pathlib import Path
from iree.compiler.tools import compile_str, compile_file

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.logger_conf import configure_logger  # noqa: E402

log = configure_logger()


class IREECompiler:
    @staticmethod
    def compile(mlir, target, opt_level, extra_args, output_file=None):
        if not os.path.exists(output_file):
            os.mkdir(output_file)
        extra_args.append(f'--iree-opt-level=O{opt_level}')
        compile_func = compile_file if os.path.isfile(mlir) else compile_str
        return compile_func(mlir, target_backends=[target], extra_args=extra_args, output_file=output_file)


def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mlir',
                        help='Path to an .mlir file with a model.',
                        required=True,
                        type=str)
    parser.add_argument('-tb', '--target_backend',
                        help='Target backend, for example "llvm-cpu" for CPU.',
                        required=True,
                        type=str)
    parser.add_argument('--opt_level',
                        help='The optimization level of the task extractions.',
                        type=int,
                        choices=[0, 1, 2, 3],
                        default=2)
    parser.add_argument('--extra_args',
                        help='The extra arguments for compilation.',
                        type=str,
                        nargs=argparse.REMAINDER,
                        default=[])
    parser.add_argument('-o', '--output_file',
                        help='Path to compiled model.',
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args


def create_dict_for_compilation(args):
    dictionary = {
        'mlir': args.mlir,
        'target_backend': args.target_backend,
        'opt_level': args.opt_level,
        'extra_args': args.extra_args,
        'output_file': args.output_file,
    }
    return dictionary


def main():
    args = cli_argument_parser()
    try:
        IREECompiler.compile(args.mlir, args.target_backend, args.opt_level, args.extra_args, args.output_file)
        if os.path.exists(args.output_file):
            print(f'The MLIR has been sucessfully compiled into {args.output_file}')
    except Exception:
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main() or 0)
