"""MLflow setup and artifact management utilities for YOLO training."""

from datetime import datetime
import os
import shutil
from pathlib import Path
from ultralytics import settings
from glob import glob


class ModeArtifacts:
    """Mode for handling artifacts: copy or move."""

    COPY = "copy"
    MOVE = "move"


class Mlflow_setup:
    """Configures MLflow tracking and manages training artifact storage."""

    ARTIFACTS_PATH = "/wyolo/worker/train_service_results"
    SUMMARY_PATH = "/wyolo/worker/events/progress.yaml"
    STOP_TRAIN_PATH = "/wyolo/worker/events/stop_training.yaml"

    COPY_MOVE = (
        ModeArtifacts.COPY
    )  # Change to ModeArtifacts.COPY to copy files instead of moving

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

    EXTENSION_PERMIT = [
        ".pt",
        ".csv",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".yaml",
        ".yml",
        ".txt",
        ".log",
    ]

    def set_config_vars(self, config: dict):
        """Sets environment variables and initializes MLflow tracking parameters."""
        self.config = config

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

            # Ensure the experiment exists and is not deleted, otherwise the
            # ultralytics MLflow callback fails with a REST exception when
            # calling set_experiment, which aborts on_train_end and leaves the
            # artifacts (e.g. evaluation_metrics/results.csv) unpublished.
            try:
                import mlflow
                from mlflow.tracking import MlflowClient

                experiment_name = config.get("sweeper", {}).get(
                    "study_name", "default_experiment"
                )
                client = MlflowClient()
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment is not None and experiment.lifecycle_stage == "deleted":
                    client.restore_experiment(experiment.experiment_id)
                    print(
                        f"--- [MLFLOW] Restored deleted experiment '{experiment_name}' ---"
                    )
                elif experiment is None:
                    mlflow.create_experiment(experiment_name)
                    print(f"--- [MLFLOW] Created experiment '{experiment_name}' ---")
            except Exception as exc:
                print(f"--- [MLFLOW] Could not prepare experiment: {exc} ---")

            os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
        else:
            settings.update({"mlflow": False})

        # Reset settings to default values
        settings.reset()

    def get_tags(self, config: dict, trainer):
        """Constructs list of experiment metadata tags for MLflow tracking."""
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

        base_tags_list.append(
            ("author", config.get("metadata", {}).get("author", "NA"))
        )

        base_tags_list.append(("experiment_type", trainer.model._get_name()))

        base_tags_list.append(
            ("version", config.get("sweeper", {}).get("version", "NA"))
        )

        base_tags_list.append(
            ("data_source", config.get("train", {}).get("data", "NA"))
        )

        for other_metadata in self.worker_metadata:
            tag_metadata = os.environ.get(other_metadata, None)
            if tag_metadata:
                base_tags_list.append((other_metadata, tag_metadata))

        return base_tags_list

    def get_summary(self, config, trainer, elapsed_time):
        """Generates real-time training progress dictionary for monitoring."""
        metadata = {}

        if "minio" in config and "mlflow" in config:
            metadata = {
                other_metadata: os.environ.get(other_metadata, None)
                for other_metadata in self.worker_metadata
            }

            trial_number = config.get("trial_number", 1)
            sweeper_config = config.get("sweeper")
            if isinstance(sweeper_config, dict):
                total_trails = sweeper_config.get("n_trials", 10)
            else:
                total_trails = 10

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

    def artifacts_organice(self, trainer=None):
        """Organizes output training artifacts and models into structured directories."""
        organized_dirs = [
            Path(self.ARTIFACTS_PATH) / "evaluation_metrics",
            Path(self.ARTIFACTS_PATH) / "model_weights",
            Path(self.ARTIFACTS_PATH) / "training_artifacts",
            Path(self.ARTIFACTS_PATH) / "training_examples",
            Path(self.ARTIFACTS_PATH) / "training_results",
            Path(self.ARTIFACTS_PATH) / "validation_examples",
        ]

        for dir_path in organized_dirs:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)

        # Write/Update results.json with current fitness value if available
        if trainer is not None:
            try:
                import json
                from slugify import slugify

                fitness_key = self.config.get("sweeper", {}).get(
                    "fitness", "metrics/accuracy_top1"
                )
                accuracy = 0.0

                if hasattr(trainer, "metrics") and trainer.metrics:
                    if fitness_key in trainer.metrics:
                        accuracy = float(trainer.metrics[fitness_key])
                    else:
                        slugified_fitness = slugify(fitness_key)
                        metrics = {
                            slugify(k): float(v) for k, v in trainer.metrics.items()
                        }
                        if slugified_fitness in metrics:
                            accuracy = metrics[slugified_fitness]
                        else:
                            # Robust fallback matching
                            matching_keys = [
                                k
                                for k, v in trainer.metrics.items()
                                if fitness_key in k or k in fitness_key
                            ]
                            if matching_keys:
                                accuracy = float(trainer.metrics[matching_keys[0]])
                                print(
                                    f"--- [ARTIFACTS] Metric '{fitness_key}' not found directly. Using '{matching_keys[0]}': {accuracy} ---"
                                )
                            else:
                                for k, v in metrics.items():
                                    if "accuracy" in k or "top1" in k:
                                        accuracy = v
                                        break

                results_file_path = os.path.join(self.ARTIFACTS_PATH, "results.json")
                if accuracy > 0.0:
                    with open(results_file_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "accuracy": round(float(accuracy), 4),
                                "status": "provisional",
                            },
                            f,
                            indent=4,
                        )
            except Exception as e:
                print(
                    f"Failed to write results.json during artifacts organization: {e}"
                )

        # Get the training results directory
        results_dir = self.config.get("tempfile")

        patron = os.path.join(results_dir, "**/*")

        for original_file in glob(patron, recursive=True):
            # Check if the file has a permitted extension
            extension = os.path.splitext(original_file)[1]
            if extension not in self.EXTENSION_PERMIT:
                continue

            if os.path.isfile(original_file):
                try:
                    # Determine target directory based on file type/name
                    if original_file.endswith(".pt") and (
                        "best" in original_file.lower()
                        or "last" in original_file.lower()
                    ):
                        destinity_dir = Path(self.ARTIFACTS_PATH) / "model_weights"
                    elif (
                        original_file.endswith(".csv")
                        or "results" in original_file.lower()
                        or "confusion_matrix" in original_file.lower()
                    ):
                        destinity_dir = Path(self.ARTIFACTS_PATH) / "evaluation_metrics"
                    elif original_file.endswith((".png", ".jpg", ".jpeg", ".svg")) and (
                        "sample" in original_file.lower()
                        or "batch" in original_file.lower()
                    ):
                        destinity_dir = Path(self.ARTIFACTS_PATH) / "training_examples"
                    elif original_file.endswith((".png", ".jpg", ".jpeg", ".svg")) and (
                        "val" in original_file.lower()
                        or "validation" in original_file.lower()
                    ):
                        destinity_dir = (
                            Path(self.ARTIFACTS_PATH) / "validation_examples"
                        )
                    elif (
                        original_file.endswith((".yaml", ".yml", ".txt", ".log"))
                        or "args" in original_file.lower()
                        or "config" in original_file.lower()
                    ):
                        destinity_dir = Path(self.ARTIFACTS_PATH) / "training_artifacts"
                    else:
                        destinity_dir = Path(self.ARTIFACTS_PATH) / "training_results"

                    if self.COPY_MOVE == ModeArtifacts.COPY:
                        shutil.copy2(str(original_file), str(destinity_dir))
                    else:
                        shutil.move(str(original_file), str(destinity_dir))
                except Exception as copy_error:
                    print(copy_error)
