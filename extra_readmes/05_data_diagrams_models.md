# Data Diagrams & Models

## 1. Configuration Data Model

The primary data structure driving the microservice is the Configuration Dictionary (typically parsed from YAML). It acts as the ultimate source of truth for the entire run.

### Entity-Relationship Representation

```mermaid
erDiagram
    CONFIGURATION ||--|| TRAINING_PARAMS : contains
    CONFIGURATION ||--o| MLFLOW_SETTINGS : configures
    CONFIGURATION ||--o| MINIO_SETTINGS : configures
    CONFIGURATION ||--o| SWEEPER_SETTINGS : orchestrates

    CONFIGURATION {
        string model_name "e.g., yolov8n.pt"
        string type "detect, segment, etc."
        string task_id "UUID for the run"
        string experiment_type
        string timestamp
    }

    TRAINING_PARAMS {
        string data "Path to dataset YAML"
        int epochs
        int imgsz
        int batch "-1 for auto"
        string device "0, 1, or cpu"
        boolean verbose
    }

    MLFLOW_SETTINGS {
        string uri "Tracking Server URL"
        string experiment_name
    }

    MINIO_SETTINGS {
        string endpoint
        string bucket
        string access_key
        string secret_key
    }
```

## 2. In-Memory State Model (Pipeline Runtime)

During execution, the pipeline engine passes a state dictionary between functions. This payload dynamically expands.

```mermaid
classDiagram
    class PipelineState {
        +dict user_config_train
        +int gpu_status
        +int dataset_status
        +int minio_status
        +dict train_results
        +string error
    }

    class TrainResults {
        +string save_dir
        +dict metrics
        +string model_path
    }

    PipelineState "1" *-- "0..1" TrainResults : populates
```

## 3. MLOps Artifact Schema

When the system logs to MLflow, it adheres to a strict schema to ensure compatibility with model registries:
-   **Artifact Path**: `/models/study_name/type/task_id/trial/`
-   **Metrics**: Keys are strictly sanitized (e.g., `metrics/mAP50-95(B)` becomes `metrics-map50-95-b`).
-   **Tags**: Hardware telemetry (GPU Name, VRAM) is injected as key-value tags associated with the run UUID.