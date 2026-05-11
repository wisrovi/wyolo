from datetime import datetime
import os
from typing import List
from ultralytics import settings


class Mlflow_setup:
    ARTIFACTS_PATH = "/wyolo/worker/train_service_results"
    SUMMARY_PATH = "/wyolo/worker/events/progress.yaml"
    STOP_TRAIN_PATH = "/wyolo/worker/events/stop_training.yaml"

    worker_metadata = [
        "debug",
        "USER",
        "WORKER_HOST",
        "WORKER_HOSTNAME",
        "WORKER_OS",
        "WORKER_KERNEL_VERSION",
        "WORKER_CPU_CORES",
        "WORKER_GATEWAY",
        "WORKER_NETWORK_INTERFACE",
        "WORKER_DOCKER_VERSION",
        "WORKER_APP_BASE_PATH",
        "WORKER_APP_ENV",
        "WORKER_HOME_DIR",
        "WORKER_CURRENT_DATE",
        "WORKER_CURRENT_TIME",
        "WORKER_GPU_COUNT",
        "WORKER_GPU_MODEL",
        "WORKER_GPU_MEMORY",
    ]

    def set_config_vars(self, config: dict):
        if "minio" in config and "mlflow" in config:
            settings.update({"mlflow": True})

            # Configurar las variables de entorno necesarias para MLflow
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = config["minio"]["MINIO_ENDPOINT"]
            os.environ["AWS_ACCESS_KEY_ID"] = config["minio"]["MINIO_ID"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = config["minio"]["MINIO_SECRET_KEY"]
            os.environ["MLFLOW_TRACKING_URI"] = config["mlflow"][
                "MLFLOW_TRACKING_URI"
            ]  # URI del servidor MLflow
            os.environ["MLFLOW_ARTIFACT_URI"] = (
                "s3://mlflow-artifacts/"  # Bucket en MinIO
            )

            # Configurar el nombre del experimento y el nombre de la ejecución
            os.environ["MLFLOW_EXPERIMENT_NAME"] = config.get("sweeper").get(
                "study_name"
            )
            os.environ["MLFLOW_RUN_NAME"] = config.get("task_id")
        else:
            settings.update({"mlflow": False})

        # Reset settings to default values
        settings.reset()

    def get_tags(self, config: dict, trainer):
        base_tags_list = list()

        base_tags_list.append(
            ("mlflow.note.content", config.get("metadata", {}).get("content"))
        )

        base_tags_list.append(
            (
                "documentation",
                config.get("metadata", {}).get("documentation", "NA"),
            )
        )

        base_tags_list.append(("author", config.get("metadata", {}).get("author", "NA")))

        base_tags_list.append(("experiment_type", trainer.model._get_name()))

        base_tags_list.append(("version", config.get("sweeper", {}).get("version", "NA")))

        base_tags_list.append(("data_source", config.get("train", {}).get("data", "NA")))

        for other_metadata in self.worker_metadata:
            tag_metadata = os.environ.get(other_metadata, None)
            if tag_metadata:
                base_tags_list.append((other_metadata, tag_metadata))

        return base_tags_list

    def get_summary(self, config, trainer, elapsed_time):
        metadata = {}

        if "minio" in config and "mlflow" in config:
            metadata = {
                other_metadata: os.environ.get(other_metadata, None)
                for other_metadata in self.worker_metadata
            }

            trial_number = config.get("trial_number", 1)
            total_trails = config.get("sweeper", 1).get("n_trials", 10)

            total_epochs = config.get("train", {}).get("epochs", 10)

            epoch = trainer.epoch + 1

            task_id = config.get("task_id", "noTaskId")

            metadata["TRIAL_NUMBER"] = trial_number
            metadata["TOTAL_TRIALS"] = total_trails
            metadata["TOTAL_EPOCHS"] = total_epochs
            metadata["EPOCH"] = epoch
            metadata["EPOCH_PROGRESS"] = epoch / total_epochs
            metadata["TRIAL_PROGRESS"] = trial_number / total_trails
            metadata["datetime"] = datetime.now().isoformat()
            metadata["task_id"] = task_id
            metadata["DEBUG_MODE"] = config.get("debug", "False")
            metadata["elapsed_time"] = round(elapsed_time, 3)

            epoch_count_total = total_epochs * int(total_trails + 1)
            epoch_total_now = int(trial_number + 1) * epoch
            metadata["result_time"] = round(
                elapsed_time * epoch_count_total / epoch_total_now, 3
            )

        return metadata

    def artifacts_organice(self):
        pass