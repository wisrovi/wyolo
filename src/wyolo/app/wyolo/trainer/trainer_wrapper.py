import sys
import os
import time
from datetime import datetime
from typing import List
import shutil
import uuid
import torch

import mlflow
from slugify import slugify
from ultralytics import RTDETR, YOLO
from ultralytics.utils.autobatch import autobatch
import yaml
from .gpu_utils import obtener_info_gpu_json
from .cte.elemental import Elemental
from .utils.mlflow_setup import Mlflow_setup
from .dto.model_wrapper import MLflowYOLOModel
from .gpu_utils import gpu_compatibility_check


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
            # Log the model as artifact instead (simpler approach)
            try:
                import torch

                model_path = f"{self.ARTIFACTS_PATH}/pytorch_model.pth"
                torch.save(pytorch_model.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path="model")
            except Exception as e:
                print(f"Failed to log model: {e}")

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

            self.artifacts_organice()
            mlflow.log_artifacts(self.ARTIFACTS_PATH)

            current_run = mlflow.active_run()
            self.model_uri = f"runs:/{current_run.info.run_id}/model"

        self.start_time = time.time()

    def on_epoch_end(self, trainer):
        if self.firts_epoch:
            # self.firts_epoch = False
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

    def tune(self, CONFIG_TRAIN: dict, _generations: int):
        if self.model:

            _grace_period = self.config.get("genetic", {}).get("min_epochs_by_ind", 10)
            _other_parameters = self.config.get("genetic", {}).get(
                "other_parameters", {}
            )

            _use_genetic = self.config.get("genetic", {}).get("use_genetic", False)
            if _use_genetic:
                # genetic algorithm
                _use_ray = False
                # Eficiencia: Lento (espera a terminar cada generación).
                # Variación: Mutación: Altera valores previos.
                # Herencia: Los hijos heredan rasgos de los padres.
                # Selección: Los mejores "padres" sobreviven.

                _other_parameters["optimizer"] = "SGD"  # Recomendado para evolución
            else:
                # ASHA (Asynchronous Successive Halving Algorithm)
                # "Pruning" (Poda) de ensayos prematuros utilizando grace_period
                _use_ray = True
                # Eficiencia: Rápido (interrumpe ensayos prematuros).
                # Variación: Muestreo: Elige valores nuevos del espacio definido.
                # Herencia: Cada experimento suele ser independiente.
                # Selección: Los peores se detienen (Poda).

                # Ensure grace_period is positive and reasonable
                _epochs = self.config.get("train", {}).get("epochs", 10)

                _other_parameters["grace_period"] = max(1, min(_epochs, _grace_period))

            # si algun parametros de _other_parameters ya existe en config_train
            # se borra para evitar conflictos
            _real_other_parameters = _other_parameters.copy()
            for key in _other_parameters.keys():
                if key in CONFIG_TRAIN:
                    _real_other_parameters.pop(key)

            _iterations = max(2, min(100, _generations))

            return self.model.tune(
                **CONFIG_TRAIN,
                iterations=_iterations,
                use_ray=_use_ray,
                **_real_other_parameters,
            )

    def train(self, config_train: dict):
        if self.model:
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

    results_dir = request_config.get("tempfile")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

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
    _data = request_config["train"].get("data", None)
    if _data is None or not os.path.exists(_data):
        raise FileNotFoundError(f"Data path not found: {_data}")

    # Check for force_gpu in extras, default to False
    force_gpu = request_config.get("extras", {}).get("force_gpu", False)
    force_cpu = request_config.get("extras", {}).get("force_cpu", False)

    if force_cpu:
        print("Forcing CPU usage for training.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        torch.cuda.is_available = lambda: False

        request_config["train"]["device"] = "cpu"

    elif gpu_compatibility_check(force_gpu):
        request_config["train"]["device"] = "0"
    else:
        request_config["train"]["device"] = "cpu"

    final_result = 0.0

    if "train" in request_config:

        if trainer.config.get("genetic", {}).get("activate", False):
            # 0. Validacion que existan los parametros necesarios
            NEED_PARAMS = [
                "poblation_size",
                "generations",
                "min_epochs_by_ind",
                "direction",
                "fitness",
            ]
            for param in NEED_PARAMS:
                if param not in trainer.config.get("genetic", {}):
                    raise ValueError(
                        f"Missing genetic parameter: {param} in configuration."
                    )

            _poblation_size = trainer.config.get("genetic", {}).get(
                "poblation_size", 10
            )
            _generations = trainer.config.get("genetic", {}).get("generations", 10)

            start_time = datetime.now().isoformat()

            # 1. Iniciar proceso de tuning genético
            # ---------------------------------------
            # ---------------------------------------
            #      Proceso de evolución genética
            # ---------------------------------------
            # ---------------------------------------
            _real_generations = _generations * _poblation_size
            tune_results = trainer.tune(
                CONFIG_TRAIN=request_config["train"],
                _generations=_real_generations,
            )
            # ---------------------------------------
            # ---------------------------------------

            # 2. Guardar resultados del tuning
            _use_genetic = trainer.config.get("genetic", {}).get("use_genetic", False)
            if _use_genetic:
                # genetic algorithm
                try:
                    experiment_path = tune_results.experiment_path
                    shutil.copytree(
                        experiment_path,
                        trainer.ARTIFACTS_PATH + "/tune_results",
                        dirs_exist_ok=True,
                    )
                except Exception as e:
                    print(f"Error copying tune results: {e}")
            else:
                # ASHA (by ray tune)
                experiment_path = tune_results.experiment_path
                shutil.copytree(
                    experiment_path,
                    trainer.ARTIFACTS_PATH + "/tune_results",
                    dirs_exist_ok=True,
                )

            # 3. Obtener el mejor resultado basado en una métrica (ejemplo: accuracy)
            _mode = trainer.config.get("genetic", {}).get("direction", "max")
            _metric = trainer.config.get("genetic", {}).get("fitness", fitness)
            best_result = tune_results.get_best_result(metric=_metric, mode=_mode)

            # 4. Extraer los parámetros (config) del mejor resultado
            best_params = best_result.config

            # 5. Actualizar request_config con los mejores parámetros
            request_config["train"].update(best_params)

            # 6. Guardar los mejores parámetros en archivo YAML
            with open(
                trainer.ARTIFACTS_PATH + "/tune_results" + "/best_params.yaml", "w"
            ) as f:
                yaml.dump(request_config["train"], f)

            # 7. Guardar duración del proceso de tuning en archivo YAML
            with open(
                trainer.ARTIFACTS_PATH + "/tune_results" + "/duration.yaml", "w"
            ) as f:
                final_result = best_result.metrics.get(_metric, 0.0)
                end_time = datetime.now().isoformat()
                yaml.dump(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": str(
                            datetime.fromisoformat(end_time)
                            - datetime.fromisoformat(start_time)
                        ),
                        "best_fitness": final_result,
                    },
                    f,
                )

        # ---------------------------------------
        # ---------------------------------------
        #     Proceso de entrenamiento normal
        # ---------------------------------------
        # ---------------------------------------
        train_params = request_config["train"]
        results = trainer.train(config_train=train_params)
        # ---------------------------------------
        # ---------------------------------------

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
