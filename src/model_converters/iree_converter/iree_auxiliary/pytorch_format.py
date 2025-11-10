import importlib
import subprocess
from converter import IREEConverter


class IREEConverterPyTorchFormat(IREEConverter):
    def __init__(self, args):
        super().__init__(args)
        self.torch = importlib.import_module('torch')
        self.aot = importlib.import_module('iree.turbine.aot')
        self.model_path = args.get('model_path', None)
        self.model_weights = args.get('model_weights', None)
        self.module = args.get('torch_module', None)
        self.input_shape = args.get('input_shape', None)

    @property
    def source_framework(self):
        return 'PyTorch'

    def __get_model_from_path(self):
        self.log.info(f'Loading model from path {self.model_path}')
        file_type = self.model_path.split('.')[-1]
        supported_extensions = ['pt']
        if file_type not in supported_extensions:
            raise ValueError(f'The file type {file_type} is not supported')
        model = self.torch.load(self.model_path)
        model.eval()
        return model

    def __get_model_from_module(self):
        self.log.info(f'Loading model {self.model_name} from module')
        model_cls = importlib.import_module(self.module).__getattribute__(self.model_name)
        if self.model_weights is None or self.model_weights == '':
            self.log.info('Loading pretrained model')
            model = model_cls(weights=True)
        else:
            self.log.info(f'Loading model with weights from file {self.model_weights}')
            model = model_cls()
            checkpoint = self.torch.load(self.model_weights, map_location=self.device.lower())
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
        return model

    def _convert_model_from_framework(self):
        model = None
        if self.module:
            model = self.__get_model_from_module()
        else:
            model = self.__get_model_from_path()
        example_arg = self.torch.randn(*self.input_shape)
        export_output = self.aot.export(model, example_arg)
        export_output.save_mlir(self.output_mlir)
        return
