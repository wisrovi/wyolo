import os
import time
import random
from datetime import datetime
from glob import glob
from typing import Dict, Any
from PIL import Image
from ultralytics import settings
from ultralytics.models import RTDETR, YOLO
from loguru import logger

from .gpu_utils import obtener_info_gpu_json, get_better_batch
from .mlflow_manager import MLflowManager
from .utils import EDAManager, ProgressManager, StatusEDA


class TrainerWrapper:
    """
    Simplified trainer wrapper with separated concerns.
    """

    config = {}
    GPU_USE = 0.4  # percentage of GPU usage
    is_configured = False
    model = None
    start_time = 0
    end_time = 0
    firts_epoch = True  # Keep original spelling for compatibility
    path_results = ""  # Store results path for EDA

    # Worker metadata for compatibility
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

    def __init__(self, config: dict):
        self.config = config
        self.hash_manager = None
        self.status_eda_completed = 2  # StatusEDA.SAVED

        # Initialize managers
        self.mlflow_manager = MLflowManager(config)
        self.eda_manager = EDAManager(config)
        self.progress_manager = ProgressManager(config)

        # Update ultralytics settings and configure system metrics
        if "minio" in self.config and "mlflow" in self.config:
            # Configure system metrics globally BEFORE ultralytics starts
            try:
                import mlflow

                mlflow.set_system_metrics_sampling_interval(5)
                mlflow.set_system_metrics_samples_before_logging(3)
                print("✅ System metrics configured globally")
            except Exception as e:
                print(f"⚠️ Could not configure system metrics: {e}")

            settings.update({"mlflow": True})
        else:
            settings.update({"mlflow": False})
        settings.reset()

        self._setup_environment()
        self._setup_redis()

    def _setup_redis(self):
        """Setup Redis connection if configured."""
        if "redis" in self.config:
            redis_config = self.config.get("redis", {})
            from wredis.hash import RedisHashManager

            self.hash_manager = RedisHashManager(
                host=redis_config.get("REDIS_HOST"),
                port=redis_config.get("REDIS_PORT"),
                db=redis_config.get("REDIS_DB"),
                verbose=False,
            )

    def _setup_environment(self):
        """Setup environment variables for MLflow and MinIO."""
        if "minio" in self.config and "mlflow" in self.config:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config["minio"][
                "MINIO_ENDPOINT"
            ]
            os.environ["AWS_ACCESS_KEY_ID"] = self.config["minio"]["MINIO_ID"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.config["minio"][
                "MINIO_SECRET_KEY"
            ]
            os.environ["MLFLOW_TRACKING_URI"] = self.config["mlflow"][
                "MLFLOW_TRACKING_URI"
            ]
            os.environ["MLFLOW_ARTIFACT_URI"] = "s3://mlflow-artifacts/"

            # Configure experiment name and run name
            experiment_name = self.config.get("sweeper", {}).get(
                "study_name", "default_experiment"
            )
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
            os.environ["MLFLOW_RUN_NAME"] = self.config.get("task_id", "default_run")

            # Set MLflow experiment
            try:
                import mlflow

                mlflow.set_experiment(experiment_name)
                print(f"MLflow experiment set: {experiment_name}")
            except Exception as e:
                print(f"Error setting MLflow experiment: {e}")

    def create_model(
        self, model_path: str = "", model_type: str = "yolo", model_name: str = ""
    ):
        """Create model instance."""
        model_to_use = model_path or model_name
        if not model_to_use:
            raise ValueError("Either model_path or model_name must be provided")

        if model_type == "yolo":
            self.model = YOLO(model_to_use)
        elif model_type == "rtdetr":
            self.model = RTDETR(model_to_use)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return self.model

    def get_better_batch(self, batch_to_use: int = 32):
        """Get optimal batch size for GPU."""
        return get_better_batch(self, batch_to_use)

    @property
    def config_train(self):
        """Get training configuration."""
        return self.config

    @config_train.setter
    def config_train(self, new_config: dict):
        """Set training configuration."""
        self.config = new_config

    def set_config_vars(self):
        """Set configuration variables (for compatibility)."""
        self._setup_environment()
        self.is_configured = True

    def obtener_info_gpu_json(self):
        """Get GPU information."""
        return obtener_info_gpu_json()

    def save_eda(self):
        """Save EDA results."""
        if self.status_eda_completed != 2:  # StatusEDA.SAVED
            if hasattr(self, "path_results"):
                self.eda_manager.save_eda(self.path_results)
                self.status_eda_completed = 2  # StatusEDA.SAVED

    def on_train_start(self, trainer):
        """Called when training starts."""
        self.start_time = getattr(trainer, "start_time", time.time())

        # Setup dataset logging
        data_path = self.config.get("train", {}).get("data", "")
        dvc_path = self.config.get("dvc_data_path", "")
        self.mlflow_manager.setup_dataset_logging(data_path, dvc_path)

        # Log system info
        gpu_info = self.obtener_info_gpu_json()
        self.mlflow_manager.log_system_info(gpu_info)

        # Log metadata
        metadata = self.config.get("metadata", {})
        metadata.update(
            {
                "experiment_type": "ClassificationModel",
                "version": "1",
                "data_source": data_path,
            }
        )
        self.mlflow_manager.log_metadata(metadata)

        # Log hyperparameters
        self.mlflow_manager.log_hyperparameters(self.config.get("train", {}))

        # Log configuration file
        self.mlflow_manager.log_config_file(self.config["config_path"])

        # Log worker metadata
        self._log_worker_metadata()

        # Upload example images
        self.log_example_images(model_type=trainer.model._get_name())

    def on_train_end(self, trainer):
        """Called when training ends."""
        self.end_time = time.time()

        # Log model and metrics
        self.mlflow_manager.log_model_and_metrics(trainer)

        # Save EDA
        self.save_eda()

        # Update progress
        final_results = {
            "fitness": trainer.metrics.get("fitness", 0),
            "accuracy": trainer.metrics.get("metrics/accuracy_top1", 0),
        }
        self.progress_manager.complete_training(final_results)

    def on_train_epoch_end(self, trainer):
        """Called at the end of each epoch."""
        if hasattr(trainer, "epoch"):
            metrics = {
                "loss": trainer.metrics.get("train/loss", 0),
                "accuracy": trainer.metrics.get("metrics/accuracy_top1", 0),
            }
            self.progress_manager.update_progress(trainer.epoch, metrics)

        # Check for stop training condition
        self.check_stop_training(trainer)

    def on_epoch_end(self, trainer):
        """Called at the end of each epoch (compatibility)."""
        # Handle first epoch logic
        if self.firts_epoch:
            self.firts_epoch = False
            try:
                import mlflow

                mlflow.log_artifacts(
                    self.config["train"]["project"] + "train_" + self.config["task_id"]
                )
            except:
                pass

        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time

        if "minio" in self.config and "mlflow" in self.config:
            metadata = {
                other_metadata: os.environ.get(other_metadata, None)
                for other_metadata in self.worker_metadata
            }

            trial_number = self.config.get("trial_number", 1)
            total_trails = self.config.get("sweeper", 1).get("n_trials", 10)
            total_epochs = self.config.get("train", {}).get("epochs", 10)
            epoch = getattr(trainer, "epoch", 0) + 1
            task_id = self.config.get("task_id", "noTaskId")

            metadata["TRIAL_NUMBER"] = str(trial_number)
            metadata["TOTAL_TRIALS"] = str(total_trails)
            metadata["TOTAL_EPOCHS"] = str(total_epochs)
            metadata["EPOCH"] = str(epoch)
            metadata["EPOCH_PROGRESS"] = str(epoch / total_epochs)
            metadata["TRIAL_PROGRESS"] = str(trial_number / total_trails)
            metadata["datetime"] = datetime.now().isoformat()
            metadata["task_id"] = task_id
            metadata["DEBUG_MODE"] = str(self.config.get("debug", "False"))
            metadata["elapsed_time"] = str(round(elapsed_time, 3))

            epoch_count_total = total_epochs * int(total_trails + 1)
            epoch_total_now = int(trial_number + 1) * epoch
            metadata["result_time"] = round(
                elapsed_time * epoch_count_total / epoch_total_now, 3
            )

            gpu_json_list = self.obtener_info_gpu_json()
            import json

            gpu_data = json.loads(gpu_json_list)
            if isinstance(gpu_data, list):
                for gpu_json in gpu_data:
                    for key, value in gpu_json.items():
                        metadata[key] = value

            redis_key = "progress" + f":{task_id}"

            if self.hash_manager:
                for metadata_key, metadata_value in metadata.items():
                    if metadata_value is not None:
                        try:
                            self.hash_manager.create_hash(
                                key=redis_key,
                                hash_name=metadata_key,
                                value=metadata_value,
                                ttl=1200,
                            )
                        except:
                            pass

            self.save_eda()

        self.check_stop_training(trainer)

    def train(self, config_train: Dict[str, Any]):
        """Start training with given configuration."""
        if not self.model:
            raise ValueError("Model not created. Call create_model() first.")

        # Check for tune parameter
        tune = self.config.get("sweeper", {}).get("tune", False)
        if tune:
            if isinstance(tune, bool):
                tune = 1
            elif isinstance(tune, int):
                tune = max(1, tune)
                tune = min(tune, 100)
            else:
                raise ValueError("Invalid tune parameter")

            grace_period = self.config.get("sweeper", {}).get("grace_period", 10)
            epochs = self.config.get("train", {}).get("epochs", 10)
            grace_period = max(1, min(epochs, grace_period))

            return self.model.tune(
                **config_train,
                iterations=tune,
                use_ray=True,
                grace_period=grace_period,
            )
        else:
            # Handle MAX_GPU environment variable
            import os

            MAX_GPU = float(os.environ.get("MAX_GPU", -5000.1))
            config_train["batch"] = max(MAX_GPU / 100, -1)

            # Set callbacks
            self.model.add_callback("on_train_start", self.on_train_start)
            self.model.add_callback("on_train_end", self.on_train_end)
            self.model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
            self.model.add_callback("on_fit_epoch_end", self.on_epoch_end)

            # Store results path for EDA
            self.path_results = config_train.get("project", "")

            # Start MLflow run with system metrics if configured
            if "minio" in self.config and "mlflow" in self.config:
                import mlflow

                if not mlflow.active_run():
                    experiment_name = self.config.get("sweeper", {}).get(
                        "study_name", "default_experiment"
                    )
                    run_name = self.config.get("task_id", "default_run")
                    mlflow.start_run(run_name=run_name, log_system_metrics=True)
                    print(f"✅ MLflow run started with system metrics: {run_name}")

            # Start training
            results = self.model.train(**config_train)
            return results

    def check_stop_training(self, trainer):
        """Check if training should be stopped."""
        task_id = self.config.get("task_id")
        stop_training_file = f"/config/stop_training_{task_id}.txt"

        if os.path.exists(stop_training_file):
            if self.model:
                # Stop training by setting the attribute directly
                setattr(self.model, "stop_training", True)
                setattr(self.model, "stop_training", False)

            if "minio" in self.config and "mlflow" in self.config:
                # Log final model
                self.on_train_end(trainer)

                # Log artifacts
                try:
                    current_training_path = (
                        self.config["train"]["project"]
                        + "train_"
                        + self.config["task_id"]
                    )
                    import mlflow

                    mlflow.log_artifacts(current_training_path)
                except:
                    pass

                # Copy model to dataset
                try:
                    original_dataset_path = self.config["train"]["data"]
                    os.makedirs(
                        os.path.join(original_dataset_path, "weights"),
                        exist_ok=True,
                    )
                    os.system(
                        f"cp {self.config['train']['project']}train_{self.config['task_id']}/weights/last.pt {original_dataset_path}weights/last.pt"
                    )
                except:
                    logger.error(
                        f"Error copying final model to {self.config['train']['data']}weights/last.pt"
                    )

                # Log final model artifact
                import mlflow

                mlflow.log_artifact(
                    os.path.join(
                        self.config["train"]["project"],
                        "train_" + self.config["task_id"],
                        "weights",
                        "last.pt",
                    ),
                    artifact_path="model_final",
                )

                # End MLflow run
                mlflow.end_run(status="FINISHED")

            raise StopIteration("Training stopped by callback condition.")

    def log_example_images(self, model_type: str):
        """Log example images to MLflow."""
        images, labels = self.get_images_and_labels(model_type=model_type)

        for i, (label, image_path) in enumerate(zip(labels, images)):
            try:
                image = Image.open(image_path)
                import mlflow

                mlflow.log_image(image, f"example_images/{label}/image_{i}.png")
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")

    def get_images_and_labels(self, model_type: str, size: int = 5):
        """Get example images and labels from dataset."""
        images = []
        labels = []

        # Handle different model types - check if it's a classification model
        # by checking the task type or model name
        is_classification = (
            "Classification" in model_type
            or "classify" in model_type.lower()
            or self.config.get("type") == "yolo"
            and self.config.get("model", "").endswith("-cls.pt")
        )

        if is_classification:
            data_path = self.config.get("train", {}).get("data", None)
            if data_path and os.path.exists(data_path):
                train_path = os.path.join(data_path, "train")
                if os.path.exists(train_path):
                    label_list = [
                        path
                        for path in os.listdir(train_path)
                        if os.path.isdir(os.path.join(train_path, path))
                    ]
                    for label in label_list:
                        label_path = os.path.join(train_path, label)
                        image_paths = []
                        for ext in ["*.jpg", "*.jpeg", "*.png"]:
                            image_paths.extend(glob(os.path.join(label_path, ext)))

                        if image_paths:
                            selected_images = random.sample(
                                image_paths, min(size, len(image_paths))
                            )
                            for image in selected_images:
                                images.append(image)
                                labels.append(label)
        else:
            logger.warning(f"Example images not implemented for {model_type}.")

        return images, labels

    def _log_worker_metadata(self):
        """Log worker environment metadata."""
        worker_metadata = {}

        # System environment variables
        env_vars = [
            "USER",
            "WORKER_HOST",
            "WORKER_HOSTNAME",
            "WORKER_OS",
            "WORKER_KERNEL_VERSION",
            "WORKER_CPU_CORES",
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

        for var in env_vars:
            value = os.environ.get(var)
            if value:
                self.mlflow_manager.log_metadata({var: value})
