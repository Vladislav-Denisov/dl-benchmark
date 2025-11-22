import subprocess
import os
from converter import IREEConverter


class IREEConverterONNXFormat(IREEConverter):
    def __init__(self, args):
        super().__init__(args)
        self.model_path = args.get('model_path', None)
        self.onnx_opset_version = args.get('onnx_opset_version', None)
        self._validate_arguments()

    @property
    def source_framework(self):
        return 'ONNX'

    def _validate_arguments(self):
        if self.model_path is None or self.model_path == '':
            raise ValueError('The model_path parameter is required for ONNX conversion.')

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Model file not found: {self.model_path}')

        if self.onnx_opset_version is None:
            raise ValueError('The onnx_opset_version parameter is required for ONNX conversion.')

    def _convert_model_from_framework(self):
        import_args = [
            'iree-import-onnx',
            self.model_path,
            '--opset-version',
            str(self.onnx_opset_version),
            '-o',
            self.output_mlir,
        ]
        import_cmd = subprocess.list2cmdline(import_args)
        subprocess.run(import_cmd, shell=True, capture_output=True)
        return
