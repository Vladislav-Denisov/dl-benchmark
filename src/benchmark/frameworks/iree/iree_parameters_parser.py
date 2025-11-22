from ..config_parser.dependent_parameters_parser import DependentParametersParser
from ..config_parser.framework_parameters_parser import FrameworkParameters


class IREEParametersParser(DependentParametersParser):
    CONFIG_FRAMEWORK_DEPENDENT_TAG = 'FrameworkDependent'
    TAG_FUNCTION_NAME = 'FunctionName'
    TAG_INPUT_SHAPE = 'InputShape'
    TAG_LAYOUT = 'Layout'
    TAG_NORMALIZE = 'Normalize'
    TAG_MEAN = 'Mean'
    TAG_STD = 'Std'
    TAG_CHANNEL_SWAP = 'ChannelSwap'
    TAG_TARGET_BACKEND = 'TargetBackend'
    TAG_OPTIMIZATION_LEVEL = 'OptimizationLevel'
    TAG_ONNX_OPSET = 'OnnxOpsetVersion'
    TAG_EXTRA_COMPILE_ARGS = 'ExtraCompileArgs'

    def parse_parameters(self, curr_test):
        dep_parameters_tag = curr_test.getElementsByTagName(self.CONFIG_FRAMEWORK_DEPENDENT_TAG)[0]

        def _read_tag(tag_name):
            tag_nodes = dep_parameters_tag.getElementsByTagName(tag_name)
            if not tag_nodes:
                return None
            node = tag_nodes[0].firstChild
            return node.data.strip() if node else None

        return IREEParameters(
            function_name=_read_tag(self.TAG_FUNCTION_NAME),
            input_shape=_read_tag(self.TAG_INPUT_SHAPE),
            layout=_read_tag(self.TAG_LAYOUT),
            normalize=_read_tag(self.TAG_NORMALIZE),
            mean=_read_tag(self.TAG_MEAN),
            std=_read_tag(self.TAG_STD),
            channel_swap=_read_tag(self.TAG_CHANNEL_SWAP),
            target_backend=_read_tag(self.TAG_TARGET_BACKEND),
            optimization_level=_read_tag(self.TAG_OPTIMIZATION_LEVEL),
            onnx_opset_version=_read_tag(self.TAG_ONNX_OPSET),
            extra_compile_args=_read_tag(self.TAG_EXTRA_COMPILE_ARGS),
        )


class IREEParameters(FrameworkParameters):
    def __init__(self, function_name, input_shape, layout, normalize, mean, std, channel_swap,
                 target_backend, optimization_level, onnx_opset_version, extra_compile_args):
        self.function_name = None
        self.input_shape = None
        self.layout = 'NHWC'
        self.normalize = None
        self.mean = None
        self.std = None
        self.channel_swap = None
        self.target_backend = 'llvm-cpu'
        self.opt_level = '2'
        self.onnx_opset_version = None
        self.extra_compile_args = None

        if not self._parameter_is_not_none(function_name):
            raise ValueError('FunctionName is a required parameter for IREE benchmark tests.')
        self.function_name = function_name

        if not self._parameter_is_not_none(input_shape):
            raise ValueError('InputShape is a required parameter for IREE benchmark tests.')
        self.input_shape = input_shape

        if self._parameter_is_not_none(layout):
            self.layout = layout
        if self._parameter_is_not_none(normalize):
            self.normalize = normalize
        if self._parameter_is_not_none(mean):
            self.mean = mean
        if self._parameter_is_not_none(std):
            self.std = std
        if self._parameter_is_not_none(channel_swap):
            self.channel_swap = channel_swap
        if self._parameter_is_not_none(target_backend):
            self.target_backend = target_backend
        if self._parameter_is_not_none(optimization_level):
            self.opt_level = optimization_level
        if self._parameter_is_not_none(onnx_opset_version):
            self.onnx_opset_version = onnx_opset_version
        if self._parameter_is_not_none(extra_compile_args):
            self.extra_compile_args = extra_compile_args

