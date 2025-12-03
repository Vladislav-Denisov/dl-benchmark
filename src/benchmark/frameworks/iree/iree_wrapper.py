from ..config_parser.test_reporter import Test
from ..framework_wrapper import FrameworkWrapper
from ..known_frameworks import KnownFrameworks
from .iree_process import IREEProcess


class IREEWrapper(FrameworkWrapper):
    framework_name = KnownFrameworks.iree

    @staticmethod
    def create_process(test, executor, log, **kwargs):
        return IREEProcess.create_process(test, executor, log)

    @staticmethod
    def create_test(model, dataset, indep_parameters, dep_parameters):
        return Test(model, dataset, indep_parameters, dep_parameters)
