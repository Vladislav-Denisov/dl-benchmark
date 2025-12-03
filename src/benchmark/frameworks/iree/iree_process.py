from pathlib import Path

from ..processes import ProcessHandler


class IREEProcess(ProcessHandler):
    benchmark_app_name = 'iree_python_benchmark'
    launcher_latency_units = 'seconds'

    def __init__(self, test, executor, log):
        super().__init__(test, executor, log)
        self.path_to_script = Path.joinpath(self.inference_script_root, 'inference_iree.py')

    @staticmethod
    def create_process(test, executor, log):
        return IREEProcess(test, executor, log)

    def get_performance_metrics(self):
        return self.get_performance_metrics_from_json_report()

    def _fill_command_line(self):
        python = ProcessHandler.get_cmd_python_version(self._test)
        arguments = self._compose_arguments()
        return f'{python} {self.path_to_script} {arguments}'.strip()

    def _compose_arguments(self):
        model = self._test.model
        dep = self._test.dep_parameters
        indep = self._test.indep_parameters

        dataset_path = self._normalize_optional(self._test.dataset.path if self._test.dataset else None)
        model_path = self._normalize_optional(model.model)
        weights_path = self._normalize_optional(model.weight)

        command = (f'-fn {dep.function_name} -is {dep.input_shape} -ni {indep.iteration} '
                   f'--report_path {self.report_path}')

        command = self._add_optional_argument_to_cmd_line(command, '-mn', model.name)

        source_framework = self._get_source_framework(model.source_framework)
        command = self._add_optional_argument_to_cmd_line(command, '-f', source_framework)

        command = self._add_optional_argument_to_cmd_line(command, '-m', model_path)
        command = self._add_optional_argument_to_cmd_line(command, '-w', weights_path)

        module_path = self._normalize_optional(model.module)
        command = self._add_optional_argument_to_cmd_line(command, '-tm', module_path)
        command = self._add_optional_argument_to_cmd_line(command, '-i', dataset_path)
        command = self._add_optional_argument_to_cmd_line(command, '-b', indep.batch_size)

        task_type = self._resolve_task_type(model)
        command = self._add_optional_argument_to_cmd_line(command, '--task', task_type)

        time_limit = indep.test_time_limit
        command = self._add_optional_argument_to_cmd_line(command, '--time', time_limit)

        layout = self._normalize_optional(dep.layout)
        command = self._add_optional_argument_to_cmd_line(command, '--layout', layout)
        if self._parameter_is_true(dep.normalize):
            command = self._add_flag_to_cmd_line(command, '--norm')

        mean = self._normalize_optional(dep.mean)
        std = self._normalize_optional(dep.std)
        channel_swap = self._normalize_optional(dep.channel_swap)
        command = self._add_optional_argument_to_cmd_line(command, '--mean', mean)
        command = self._add_optional_argument_to_cmd_line(command, '--std', std)
        command = self._add_optional_argument_to_cmd_line(command, '--channel_swap', channel_swap)

        target_backend = self._normalize_optional(dep.target_backend) or 'llvm-cpu'
        command = self._add_optional_argument_to_cmd_line(command, '-tb', target_backend)

        opt_level = self._normalize_optional(dep.opt_level) or '2'
        command = self._add_optional_argument_to_cmd_line(command, '--opt_level', opt_level)

        onnx_opset = self._normalize_optional(dep.onnx_opset_version)
        command = self._add_optional_argument_to_cmd_line(command, '--onnx_opset_version', onnx_opset)

        if indep.raw_output:
            command = self._add_argument_to_cmd_line(command, '--raw_output', indep.raw_output)

        extra_compile_args = self._normalize_optional(dep.extra_compile_args)

        if extra_compile_args:
            command = f'{command} --extra_compile_args {extra_compile_args}'

        return command.strip()

    @staticmethod
    def _normalize_optional(value):
        if value is None:
            return None
        string_value = str(value).strip()
        if not string_value or string_value.lower() == 'none':
            return None
        return string_value

    @staticmethod
    def _parameter_is_true(value):
        if value is None:
            return False
        return str(value).strip().lower() in ['true', '1', 'yes']

    @staticmethod
    def _get_source_framework(value):
        normalized_value = IREEProcess._normalize_optional(value)
        if not normalized_value:
            return None
        normalized_value = normalized_value.lower()
        allowed_frameworks = {'onnx', 'pytorch'}
        if normalized_value in allowed_frameworks:
            return normalized_value
        return None

    @staticmethod
    def _resolve_task_type(model):
        candidate = getattr(model, 'task', None)
        normalized_candidate = IREEProcess._normalize_optional(candidate)
        if not normalized_candidate:
            return None
        normalized_candidate = normalized_candidate.lower()
        allowed_tasks = {'feedforward', 'classification'}
        if normalized_candidate in allowed_tasks:
            return normalized_candidate
        return None
