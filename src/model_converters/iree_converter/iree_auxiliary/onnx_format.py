import subprocess
from converter import IREEConverter


class IREEConverterONNXFormat(IREEConverter):
    def __init__(self, args):
        super().__init__(args)
        self.model_path = args.get('model_path', None)
        self.onnx_opset_version = args.get('onnx_opset_version', None)

    @property
    def source_framework(self):
        return 'ONNX'

    def _convert_model_from_framework(self):
        import_args = [
            "iree-import-onnx",
            self.model_path,
            "--opset-version",
            str(self.onnx_opset_version),
            "-o",
            self.output_mlir,
        ]
        import_cmd = subprocess.list2cmdline(import_args)
        ret = subprocess.run(import_cmd, shell=True, capture_output=True)
        return
