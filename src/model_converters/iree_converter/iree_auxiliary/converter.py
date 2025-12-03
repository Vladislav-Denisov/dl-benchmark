import abc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.joinpath('utils')))
from logger_conf import configure_logger  # noqa: E402

log = configure_logger()


class IREEConverter(metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.model_name = args.get('model_name', None)
        self.output_mlir = args.get('output_mlir', None)
        self.log = log

    @abc.abstractmethod
    def _convert_model_from_framework(self):
        pass

    @property
    @abc.abstractmethod
    def source_framework(self):
        pass

    @staticmethod
    def get_converter(args):
        framework = args['source_framework'].lower()
        if framework == 'onnx':
            from onnx_format import IREEConverterONNXFormat
            return IREEConverterONNXFormat(args)
        elif framework == 'pytorch':
            from pytorch_format import IREEConverterPyTorchFormat
            return IREEConverterPyTorchFormat(args)

    def convert_to_mlir(self):
        self.log.info(f'Get IREE MLIR for {self.model_name} from {self.source_framework} framework')
        self._convert_model_from_framework()
        return
