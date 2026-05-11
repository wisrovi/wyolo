"""
Example of YOLO training with MLflow and MinIO integration.
This script shows how wyolo automatically tracks metrics and registers models.
"""

from wyolo.core.trainer_wrapper import TrainerWrapper

def main():
    # Configuration with MLOps sections
    config = {
        "model": "yolov8n.pt",
        "type": "detect",
        "task_id": "exp-mlflow-001",
        "sweeper": {
            "study_name": "enterprise-detection"
        },
        "mlflow": {
            "uri": "http://localhost:5000", # Your MLflow Tracking Server
            "experiment_name": "ObjectDetection_Production"
        },
        "minio": {
            "endpoint": "localhost:9000",
            "bucket": "mlflow-artifacts",
            "access_key": "minioadmin",
            "secret_key": "minioadmin"
        },
        "train": {
            "data": "coco8.yaml",
            "epochs": 5,
            "imgsz": 640,
            "batch": -1, # Auto-batching enabled
            "device": 0  # Use GPU 0
        }
    }

    # Initialize trainer - This will automatically setup the MLflow connection
    trainer = TrainerWrapper(config=config)

    # Create model
    trainer.create_model(
        model_name=config["model"],
        model_type=config["type"]
    )

    # Start training
    # wyolo will use callbacks to log metrics, parameters, and tags to MLflow
    # At the end, it will register the model in the MLflow Model Registry
    print("--- Starting MLOps Training ---")
    trainer.train(config_train=config["train"])
    print("--- Training finished. Check your MLflow Dashboard ---")

if __name__ == "__main__":
    main()
