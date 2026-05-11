#!/usr/bin/env python3
"""
Script to register a trained model in MLflow Model Registry.
Usage: python register_model.py --run_id <run_id> --model_name <model_name>
"""

import mlflow
import argparse
import os


def register_model(run_id, model_name, artifact_path="yolo_model/best.pt"):
    """
    Register a model from a run to MLflow Model Registry.

    Args:
        run_id: MLflow run ID
        model_name: Name for the registered model
        artifact_path: Path to the model artifact in the run
    """
    try:
        # Get MLflow tracking URI from environment or use default
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)

        # Register the model
        model_uri = f"runs:/{run_id}/{artifact_path}"
        registered_model = mlflow.register_model(model_uri, model_name)

        print(f"✅ Model registered successfully!")
        print(f"   Model Name: {registered_model.name}")
        print(f"   Version: {registered_model.version}")
        print(f"   Run ID: {run_id}")
        print(f"   Model URI: {model_uri}")

        return registered_model

    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Register a model in MLflow Model Registry"
    )
    parser.add_argument("--run_id", required=True, help="MLflow run ID")
    parser.add_argument(
        "--model_name", required=True, help="Name for the registered model"
    )
    parser.add_argument(
        "--artifact_path",
        default="yolo_model/best.pt",
        help="Path to model artifact in run (default: yolo_model/best.pt)",
    )

    args = parser.parse_args()

    # Register the model
    register_model(args.run_id, args.model_name, args.artifact_path)


if __name__ == "__main__":
    main()
