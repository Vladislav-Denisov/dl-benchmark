import os
from iree.compiler.tools import compile_str, compile_file


class IREECompiler:
    @staticmethod
    def compile_model(mlir, target, opt_level, extra_args, output_file=None):
        if output_file and not os.path.exists(output_file):
            os.mkdir(output_file)
        extra_args.append(f'--iree-opt-level=O{opt_level}')
        compile_func = compile_file if os.path.isfile(mlir) else compile_str
        return compile_func(mlir, target_backends=[target], extra_args=extra_args, output_file=output_file)
