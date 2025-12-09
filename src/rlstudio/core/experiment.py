from typing import Dict, Any, Optional
import mlflow
from rlstudio.pipeline import Pipeline


class Experiment:
    """
    A self-contained unit encapsulating a Pipeline, Configuration, and Tracking context.
    """

    def __init__(
        self,
        name: str,
        pipeline: Pipeline,
        config: Dict[str, Any],
        tracking_uri: Optional[str] = None,
    ):
        self.name = name
        self.pipeline = pipeline
        self.config = config
        self.tracking_uri = tracking_uri

    def run(self):
        """
        Executes the experiment within an MLflow run context.
        """
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.name)

        with mlflow.start_run():
            # Log Params
            mlflow.log_params(self.config)

            # Log Pipeline Definition (as artifact)?
            # with open("pipeline.txt", "w") as f: f.write(str(self.pipeline))
            # mlflow.log_artifact("pipeline.txt")

            # Run Pipeline
            results = self.pipeline.run()

            # Log Metrics if any present in results
            if "metrics" in results:
                mlflow.log_metrics(results["metrics"])

            return results
