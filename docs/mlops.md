# MLOps Integration

`wyolo` is built for MLOps from the ground up.

## MLflow

By providing an `mlflow` section in your config, `wyolo` will:
1. Initialize the MLflow client.
2. Log all training parameters.
3. Log metrics (mAP, loss, etc.) at the end of training.
4. Register the model in the MLflow Model Registry.

## Artifact Management

Artifacts are automatically organized and can be synced to MinIO/S3.

## Data Versioning with DVC

Ensure your datasets are versioned using DVC to maintain full reproducibility of your experiments.
