import argparse
import os
import sys
import traceback
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.joinpath('iree_auxiliary')))
from converter import IREEConverter  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.logger_conf import configure_logger  # noqa: E402

log = configure_logger()


def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--source_framework',
                        help='Source model framework',
                        required=True,
                        type=str,
                        choices=['onnx', 'pytorch'],
                        dest='source_framework')
    parser.add_argument('-mn', '--model_name',
                        help='Model name.',
                        type=str,
                        dest='model_name')
    parser.add_argument('-m', '--model',
                        help='Path to an .onnx or .pt file with a trained model.',
                        type=str,
                        dest='model_path')
    parser.add_argument('-w', '--weights',
                        help='Path to an .pth file with a trained weights.',
                        type=str,
                        dest='model_weights')
    parser.add_argument('-tm', '--torch_module',
                        help='Torch module with model architecture.',
                        default='torchvision.models',
                        type=str,
                        dest='torch_module')
    parser.add_argument('--onnx_opset_version',
                        help='Path to an .onnx with a trained model.',
                        type=int,
                        default=18,
                        dest='onnx_opset_version')
    parser.add_argument('-is', '--input_shape',
                        help='Input shape BxCxHxW, B is a batch size,'
                             'C is an input tensor number of channels,'
                             'H is an input tensor height,'
                             'W is an input tensor width.',
                        type=int,
                        nargs=4,
                        dest='input_shape')
    parser.add_argument('-o', '--output_mlir',
                        help='Path to save the MLIR.',
                        required=True,
                        type=str,
                        dest='output_mlir')
    args = parser.parse_args()
    return args


def create_dict_for_converter(args):
    dictionary = {
        'source_framework': args.source_framework,
        'model_name': args.model_name,
        'model_path': args.model_path,
        'model_weights': args.model_weights,
        'torch_module': args.torch_module,
        'onnx_opset_version': args.onnx_opset_version,
        'input_shape': args.input_shape,
        'output_mlir': args.output_mlir,
    }
    return dictionary


def main():
    args = cli_argument_parser()
    try:
        converter = IREEConverter.get_converter(create_dict_for_converter(args))
        converter.convert_to_mlir()
        if os.path.exists(args.output_mlir):
            print(f'The MLIR has been sucessfully saved into {args.output_mlir}')
    except Exception:
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main() or 0)
