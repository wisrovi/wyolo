import os
import mlflow
import mlflow.data
import mlflow.data.filesystem_dataset_source
import mlflow.data.http_dataset_source
from mlflow import pytorch
from slugify import slugify
from ultralytics import YOLO


class MLflowManager:
    """Manages MLflow operations for model training."""

    def __init__(self, config: dict):
        self.config = config
        self.redis_manager = None
        self._setup_system_metrics()

    def _setup_system_metrics(self):
        """Setup MLflow system metrics logging."""
        try:
            # Enable system metrics collection only if not already configured
            if not hasattr(self, "_system_metrics_configured"):
                mlflow.set_system_metrics_sampling_interval(5)  # Sample every 5 seconds
                mlflow.set_system_metrics_samples_before_logging(
                    3
                )  # Log after 3 samples
                self._system_metrics_configured = True
                print("✅ System metrics logging enabled")
        except Exception as e:
            print(f"⚠️ Could not enable system metrics: {e}")
            self._system_metrics_configured = (
                False  # Mark as attempted to avoid retries
            )

    def setup_dataset_logging(self, data_path: str, dvc_path: str = ""):
        """Setup dataset logging in MLflow."""
        try:
            if dvc_path:
                try:
                    import dvc.api

                    data_url = dvc.api.get_url(dvc_path)
                    dataset_path = data_path
                except:
                    # Fallback for DVC API issues
                    dvc_path = data_path.replace("/datasets/", "")
                    data_url = f"http://{os.environ.get('CONTROL_HOST', 'localhost')}:23443/files/{dvc_path}"
                    dataset_path = data_path
            else:
                dvc_path = data_path.replace("/datasets/", "")
                data_url = f"http://{os.environ.get('CONTROL_HOST', 'localhost')}:23443/files/{dvc_path}"
                dataset_path = data_path

            if data_url:
                # Simple dataset logging for MLflow 2.20.3
                mlflow.set_tag("dataset_url", data_url)

            mlflow.set_tag("data_source", dataset_path)

        except Exception as e:
            print(f"Error logging dataset: {e}")

    def log_config_file(self, config_path: str):
        """Log configuration file to MLflow."""
        try:
            mlflow.log_artifact(config_path)
        except Exception as e:
            print(f"Error logging config file: {e}")

    def log_model_and_metrics(self, trainer):
        """Log model and metrics to MLflow."""
        if "minio" in self.config and "mlflow" in self.config:
            # Generate registered model name from experiment and task info
            experiment_name = self.config.get("sweeper", {}).get(
                "study_name", "default_experiment"
            )
            task_id = self.config.get("task_id", "default_task")
            registered_model_name = f"{experiment_name}_{task_id}"

            # Log metrics first (simple and reliable)
            try:
                metrics = {
                    slugify(key): float(value) for key, value in trainer.metrics.items()
                }
                mlflow.log_metrics(metrics)
                print("✅ Training metrics logged successfully")
            except Exception as e:
                print(f"⚠️ Error logging metrics: {e}")

            # Log model metadata
            try:
                model_metadata = {
                    "model_type": "YOLO Classification",
                    "framework": "ultralytics",
                    "registered_model_name": registered_model_name,
                    "experiment_name": experiment_name,
                    "task_id": task_id,
                }
                for key, value in model_metadata.items():
                    mlflow.set_tag(key, value)
                print("✅ Model metadata logged")
            except Exception as meta_e:
                print(f"⚠️ Error logging model metadata: {meta_e}")

            # Log model and register it properly
            try:
                import torch
                from mlflow.tracking import MlflowClient

                # Get's best model file
                best_model_path = os.path.join(
                    str(trainer.save_dir), "weights", "best.pt"
                )

                if os.path.exists(best_model_path):
                    # Log's .pt file as artifact
                    mlflow.log_artifact(best_model_path, "model")
                    print("✅ Model .pt file logged as artifact")

                    # Try to register's model using MLflow log_model
                    try:
                        best_model = YOLO(best_model_path)
                        pytorch_model = best_model.model

                        # Create a proper MLflow-compatible model inheriting from torch.nn.Module
                        class YOLOMLflowModel(torch.nn.Module):
                            def __init__(self, model):
                                super().__init__()
                                self.model = model

                            def forward(self, x):
                                return self.model(x)

                            def predict(self, data):
                                return self.model(data)

                        # Create MLflow model
                        mlflow_model = YOLOMLflowModel(pytorch_model)

                        # Log using pytorch.log_model (this creates a proper MLflow model)
                        pytorch.log_model(
                            pytorch_model=mlflow_model,
                            artifact_path="model",
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
                        print("✅ Model logged successfully with pytorch.log_model")
                        print(
                            "🎯 Model saved as proper MLflow model - check UI in Artifacts/model/"
                        )

                        # Get current run info and show model URI
                        current_run = mlflow.active_run()
                        if current_run:
                            model_uri = f"runs:/{current_run.info.run_id}/model"
                            print(f"✅ Model saved as MLflow model at: {model_uri}")
                            print(
                                f"🔍 Check MLflow UI: {mlflow.get_tracking_uri()}/#/experiments/{current_run.info.experiment_id}/runs/{current_run.info.run_id}"
                            )
                            print(
                                f"📁 Model should be visible in Artifacts section under 'model/' folder"
                            )

                            # Try to register it (will work if Model Registry is enabled)
                            try:
                                mlflow.register_model(
                                    model_uri=model_uri, name=registered_model_name
                                )
                                print(
                                    f"✅ Model registered as: {registered_model_name}"
                                )
                            except Exception as reg_e:
                                print(f"⚠️ Registration failed: {reg_e}")
                                print(f"💡 Model is available at: {model_uri}")
                                print(
                                    f"💡 Register manually: mlflow.register_model('{model_uri}', '{registered_model_name}')"
                                )

                    except Exception as log_e:
                        print(f"⚠️ PyTorch model logging failed: {log_e}")
                        print("💡 Model .pt file is still available as artifact")

                    # Fallback: log .pt file as artifact
                    if os.path.exists(best_model_path):
                        mlflow.log_artifact(best_model_path, "model")
                        print("✅ Model .pt file logged as artifact (fallback)")

                    # Try to register's model using MLflow log_model
                    try:
                        import torch

                        # Get's PyTorch model from trainer

                        pytorch_model = trainer.model.model

                        # Create a simple wrapper for MLflow compatibility
                        class YOLOModelWrapper:
                            def __init__(self, model):
                                self.model = model

                            def predict(self, data):
                                return self.model(data)

                            def __call__(self, data):
                                return self.model(data)

                        wrapped_model = YOLOModelWrapper(pytorch_model)

                        # Log's model using pytorch.log_model properly
                        try:
                            import torch

                            # Create a proper MLflow-compatible model inheriting from torch.nn.Module
                            class MLflowYOLOModel(torch.nn.Module):
                                def __init__(self, model):
                                    super().__init__()
                                    self.model = model

                                def forward(self, x):
                                    return self.model(x)

                                def predict(self, data):
                                    return self.model(data)

                            # Create the MLflow model
                            mlflow_model = MLflowYOLOModel(pytorch_model)

                            # Log using pytorch.log_model (this creates a proper MLflow model)
                            pytorch.log_model(
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
                            print("✅ Model logged successfully with pytorch.log_model")

                            # Get current run info and show model URI
                            current_run = mlflow.active_run()
                            if current_run:
                                model_uri = f"runs:/{current_run.info.run_id}/model"
                                print(f"✅ Model saved as MLflow model at: {model_uri}")
                                print(
                                    f"🔍 Check MLflow UI at: {mlflow.get_tracking_uri()}/#/experiments/{current_run.info.experiment_id}/runs/{current_run.info.run_id}"
                                )
                                print(
                                    f"📁 Model should be visible in Artifacts section under 'model/' folder"
                                )

                        except Exception as log_e:
                            print(f"⚠️ pytorch.log_model failed: {log_e}")
                            print("💡 Model .pt file is still available as artifact")

                        # Get's current run info
                        active_run = mlflow.active_run()
                        if active_run:
                            model_uri = f"runs:/{active_run.info.run_id}/pytorch_model"
                            print(f"💡 Model available at: {model_uri}")

                            # Try to register's model
                            try:
                                mlflow.register_model(
                                    model_uri=model_uri, name=registered_model_name
                                )
                                print(
                                    f"✅ Model registered as: {registered_model_name}"
                                )
                            except Exception as reg_e:
                                print(f"⚠️ Registration failed: {reg_e}")
                                print(
                                    f"💡 Model is logged and available at: {model_uri}"
                                )
                                print(
                                    f"💡 Register manually: mlflow.register_model('{model_uri}', '{registered_model_name}')"
                                )
                        else:
                            print("⚠️ No active MLflow run found")

                    except Exception as log_e:
                        print(f"⚠️ PyTorch model logging failed: {log_e}")
                        print("💡 Model .pt file is still available as artifact")
                else:
                    print("❌ Model file not found")

            except Exception as e:
                print(f"❌ Error in model logging: {e}")
                print("💡 Model .pt file is still available as artifact")

    def log_system_info(self, gpu_info: str):
        """Log system information to MLflow."""
        import json

        gpu_data = json.loads(gpu_info)

        if isinstance(gpu_data, list):
            for gpu_dict in gpu_data:
                for key, value in gpu_dict.items():
                    mlflow.set_tag(key, value)
        else:
            for key, value in gpu_data.items():
                mlflow.set_tag(key, value)

    def log_metadata(self, metadata: dict):
        """Log metadata to MLflow."""
        for key, value in metadata.items():
            mlflow.set_tag(key, value)

    def log_hyperparameters(self, config: dict):
        """Log hyperparameters to MLflow."""
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    try:
                        mlflow.log_param(f"{key}_{sub_key}", sub_value)
                    except Exception:
                        pass  # Skip if parameter already logged
            else:
                try:
                    mlflow.log_param(key, value)
                except Exception:
                    pass  # Skip if parameter already logged
