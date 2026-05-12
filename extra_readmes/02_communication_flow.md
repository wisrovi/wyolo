# Communication Flow

## 1. Internal vs External Communication

The microservice operates as a hybrid entity:
-   **Internally**: It communicates synchronously via in-memory state passing orchestrated by the pipeline engine. State dictionaries are passed from one atomic function to the next.
-   **Externally**: It relies heavily on asynchronous/RESTful APIs to validate infrastructure and log metrics. It communicates with S3/MinIO for dataset/artifact retrieval, MLflow for experiment tracking, and potentially Redis for asynchronous event broadcasting.

## 2. Cross-System Sequence Diagram

The following sequence diagram illustrates the complex flow of a successful training lifecycle, highlighting the boundaries between the local worker and external enterprise services.

```mermaid
sequenceDiagram
    autonumber
    participant Worker as Service Orchestrator
    participant State as Pipeline States
    participant Core as Trainer Wrapper
    participant Engine as DL Engine (YOLO)
    participant MLflow as MLflow Tracking Server
    participant S3 as MinIO/S3 Storage

    Worker->>State: Initiate Pipeline
    activate State
    State->>S3: Validate Bucket Connectivity
    S3-->>State: 200 OK
    State->>Worker: Validation Passed
    deactivate State

    Worker->>State: Check Dataset Integrity
    activate State
    State->>S3: Pull DVC tracked data (Optional)
    State-->>Worker: Dataset Available
    deactivate State

    Worker->>Core: Invoke Training (gpu_status==1)
    activate Core
    Core->>MLflow: Create/Fetch Experiment
    MLflow-->>Core: Experiment ID
    Core->>MLflow: Start Run & Log Hyperparameters

    Core->>Engine: Begin Epoch Loop
    activate Engine
    loop Every Epoch
        Engine-->>Core: Metrics (mAP, Loss)
        Core->>MLflow: Log Metrics Stream
    end
    Engine-->>Core: Return best.pt (Weights)
    deactivate Engine

    Core->>S3: Upload Artifacts (Graphs, Weights)
    S3-->>Core: Upload Confirmed
    Core->>MLflow: Register Model Version
    MLflow-->>Core: Model URI

    Core-->>Worker: Training Results Dictionary
    deactivate Core
    Worker->>Worker: Write Final Status to SQLite WAL
```

## 3. Communication Resilience

-   **Failure Isolation**: If communication with MLflow or S3 fails during the initialization states (`check_minio_buckets`), the pipeline halts before allocating GPU memory, routing directly to the `error_capture` state.
-   **Timeout Enforcement**: The entire pipeline is wrapped in a `TaskTimer` (e.g., 900 seconds), ensuring that network hangs with external services do not lock the worker indefinitely.