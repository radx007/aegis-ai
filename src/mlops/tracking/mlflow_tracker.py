from .base import ExperimentTracker


class MLflowTracker(ExperimentTracker):

    def start_run(self, run_name=None):
        raise NotImplementedError

    def log_parameters(self, parameters):
        raise NotImplementedError

    def log_metrics(self, metrics):
        raise NotImplementedError

    def log_artifact(self, artifact):
        raise NotImplementedError

    def log_model(self, model):
        raise NotImplementedError

    def end_run(self):
        raise NotImplementedError