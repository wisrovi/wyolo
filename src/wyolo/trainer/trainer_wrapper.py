import os
import time
from datetime import datetime
from typing import List
import shutil
import uuid

import mlflow
from slugify import slugify
from ultralytics import RTDETR, YOLO, settings
from ultralytics.utils.autobatch import autobatch
import yaml
from .gpu_utils import obtener_info_gpu_json
from .cte.elemental import Elemental
from .utils.mlflow_setup import Mlflow_setup
from .dto.model_wrapper import MLflowYOLOModel


class TrainerWrapper(Elemental, Mlflow_setup):
    # https://github.com/ultralytics/ultralytics/issues/8214
    config = {}

    def __init__(self, config: dict):
        self.config = config

        # Update a setting
        self.set_config_vars(self.config)
        self.is_configured = True

    @property
    def config_train(self):
        return self.config

    @config_train.setter
    def config_train(self, new_config: dict):
        self.config = new_config

    def on_train_end(self, trainer):
        if "minio" in self.config and "mlflow" in self.config:
            pytorch_model = trainer.model.model

            experiment_name = self.config.get("sweeper", {}).get(
                "study_name", "default_experiment"
            )
            task_id = self.config.get("task_id", "default_task")

            registered_model_name = f"{experiment_name}_{task_id}"
            # Create the MLflow model
            mlflow_model = MLflowYOLOModel(pytorch_model)
            # Log using pytorch.log_model (this creates a proper MLflow model)
            mlflow.pytorch.log_model(
                pytorch_model=mlflow_model,
                artifact_path="model",
                registered_model_name=registered_model_name,
                conda_env={
                    "channels": ["defaults", "pytorch", "conda-forge"],
                    "dependencies": [
                        "python=3.8",
                        "pytorch",
                        "torchvision",
                        "ultralytics",
                        "mlflow",
                        "numpy",
                        "pillow",
                        "pyyaml",
                    ],
                },
            )

            metrics = {
                slugify(key): float(value) for key, value in trainer.metrics.items()
            }
            mlflow.log_metrics(metrics)

            self.artifacts_organice()
            mlflow.log_artifacts(self.ARTIFACTS_PATH)

    def on_train_start(self, trainer):
        if "minio" in self.config and "mlflow" in self.config:

            # remove batch of self.config
            config_copy = self.config.copy()
            config_copy["train"].pop("batch")

            tags_list = [
                (key, value)
                for key, value in obtener_info_gpu_json()[0].items()
                if key and value
            ]

            for key, value in self.get_tags(self.config, trainer):
                tags_list.append((key, value))

            for key, value in tags_list:
                try:
                    mlflow.set_tag(key, value)
                except:
                    pass

            for key, value in config_copy["train"].items():
                try:
                    mlflow.log_param(key, value)
                except:
                    pass

            shutil.copy2(self.config["config_path"], self.ARTIFACTS_PATH)

            self.artifacts_organice()
            mlflow.log_artifacts(self.ARTIFACTS_PATH)

            current_run = mlflow.active_run()
            self.model_uri = f"runs:/{current_run.info.run_id}/model"

        self.start_time = time.time()

    def on_epoch_end(self, trainer):
        if self.firts_epoch:
            self.firts_epoch = False
            self.artifacts_organice()
            mlflow.log_artifacts(self.ARTIFACTS_PATH)

        self.end_time = time.time()

        elapsed_time = self.end_time - self.start_time

        metadata = self.get_summary(self.config, trainer, elapsed_time)

        if "minio" in self.config and "mlflow" in self.config:
            gpu_json_list: List[dict] = obtener_info_gpu_json()
            for gpu_json in gpu_json_list:
                for key, value in gpu_json.items():
                    metadata[key] = value

            with open(self.SUMMARY_PATH, "w") as file:
                yaml.dump(
                    metadata,
                    file,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

        if os.path.exists(self.STOP_TRAIN_PATH):
            self.force_stop_train(trainer)

            os.remove(self.STOP_TRAIN_PATH)

    def force_stop_train(self, trainer):

        if self.model:
            self.model.stop_training = True
            self.model.stop_training = False
            # self.model = None

        if "minio" in self.config and "mlflow" in self.config:
            # Log the final model in MLflow
            self.on_train_end(trainer)

            # Log the artifacts
            try:
                current_training_path = (
                    self.config["train"]["project"] + "train_" + self.config["task_id"]
                )
                mlflow.log_artifacts(current_training_path)
            except:
                pass

            # End the MLflow run
            mlflow.end_run(status="FINISHED")

        raise StopIteration("Entrenamiento detenido por condición de callback.")

    def train(self, config_train: dict):
        if self.model:
            tune = self.config.get("sweeper", {}).get("tune", False)
            if tune:
                if isinstance(tune, bool):
                    tune = 1
                elif isinstance(tune, int):
                    tune = max(1, tune)
                    tune = min(tune, 100)
                else:
                    raise

                grace_period = self.config.get("sweeper", {}).get("grace_period", 10)
                epochs = self.config.get("train", {}).get("epochs", 10)

                # Ensure grace_period is positive and reasonable
                grace_period = max(1, min(epochs, grace_period))

                return self.model.tune(
                    **config_train,
                    iterations=tune,
                    use_ray=True,
                    grace_period=grace_period,
                )
            else:
                # read env var MAX_GPU (user defined) for max gpu (value between 0 and 100)
                # if not set, use default value (60%),
                # if set to -1, use 60% of the gpu
                MAX_GPU = float(os.environ.get("MAX_GPU", -5000.1))

                config_train["batch"] = max(MAX_GPU / 100, -1)

                return self.model.train(**config_train)

    def create_model(self, model_name, model_type):
        if model_type == "yolo":
            model = YOLO(model_name)
        elif model_type == "rtdetr":
            model = RTDETR(model_name)
        else:
            raise ValueError("Invalid model type specified.")

        self.model = model

        # Configure the callbacks
        if "minio" in self.config and "mlflow" in self.config:
            self.model.add_callback("on_train_start", self.on_train_start)
            self.model.add_callback("on_train_end", self.on_train_end)
            self.model.add_callback("on_fit_epoch_end", self.on_epoch_end)

        return model


def get_datetime():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_trainer(config_path: str, trial_number):
    request_config = load_config(config_path=config_path)
    request_config["config_path"] = config_path

    trainer = TrainerWrapper(config=request_config)
    trainer.create_model(
        model_name=request_config["model"],
        model_type=request_config["type"],
    )

    if "task_id" not in request_config:
        request_config["task_id"] = str(uuid.uuid4())

    experiment_name = request_config.get("sweeper").get("study_name")
    tempfile = request_config.get("tempfile", "")

    RESULT_PATH = f"{tempfile}/models/{experiment_name}/{request_config['type']}/{request_config['task_id']}/"
    os.makedirs(f"{RESULT_PATH}/trail_history", exist_ok=True)
    request_config["path_results"] = f"{RESULT_PATH}/{trial_number}/"

    timestamp = get_datetime()
    request_config["timestamp"] = timestamp

    if request_config["train"]["batch"] > 0:
        better_batch = trainer.get_better_batch(
            batch_to_use=request_config["train"]["batch"]
        )
        if request_config["train"]["batch"] > better_batch:
            request_config["train"]["batch"] = better_batch

    request_config["train"]["project"] = f"{RESULT_PATH}/{trial_number}/"
    request_config["train"]["name"] = f"train_{request_config.get('task_id')}"
    request_config["train"]["verbose"] = True
    request_config["train"]["plots"] = True
    request_config["train"]["exist_ok"] = True

    trainer.config_train = request_config

    return trainer, request_config


def train(trainer: TrainerWrapper, request_config: dict, fitness: str):
    if "train" in request_config:
        results = trainer.train(config_train=request_config["train"])

        final_result = 0

        if results:
            request_config["experiment_type"] = str(results.task)
            request_config["train"]["results"] = results.results_dict

            try:
                final_result = request_config["train"]["results"][fitness]
            except:
                final_result = request_config["train"]["results"]["fitness"]

            print(f"ResultadoFinal:{final_result}")
    return final_result


if __name__ == "__main__":
    request_config: dict = ...
    trial_number = 1
    fitness: str = ...

    trainer, request_config = create_trainer(request_config, trial_number)

    if "train" in request_config:
        final_result = train(trainer, request_config)
