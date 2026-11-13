<p align="center">
  <a href="https://pypi.org/project/wyolo/"><img src="https://img.shields.io/pypi/v/wyolo?style=for-the-badge&logo=pypi&color=3b82f6" alt="PyPI version" /></a>
  <a href="https://linkedin.com/in/wisrovi-rodriguez"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  <a href="https://wisrovi.dev"><img src="https://img.shields.io/badge/Author-wisrovi.dev-111827?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Portal" /></a>
  <a href="https://orcid.org/0009-0005-0710-1861"><img src="https://img.shields.io/badge/ORCID-0009--0005--0710--1861-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License" /></a>
</p>

# 🚀 Wyolo - Professional YOLO Training Library

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Coverage](https://img.shields.io/badge/coverage-85%25-yellow.svg)]()

> **Wyolo** is a professional YOLO training library with comprehensive MLOps integration, designed for production-grade computer vision projects.

![Logo Suggestion](https://via.placeholder.com/150x150/4CAF50/FFFFFF?text=WYOLO)

---

## 📋 Project Overview

Wyolo is a sophisticated Python library that streamlines the training of YOLO (You Only Look Once) models for computer vision tasks including **object detection**, **image classification**, and **semantic segmentation**. Built on top of Ultralytics YOLO, it provides enterprise-grade features including:

- **MLOps Integration** with MLflow experiment tracking
- **GPU Optimization** with intelligent resource management
- **Hyperparameter Tuning** using genetic algorithms and ASHA
- **Container-Ready** architecture for cloud deployment
- **Comprehensive Logging** and artifact management

The library is designed for data scientists, ML engineers, and researchers who need a robust, scalable solution for training YOLO models in production environments.

---

## 🚶 Diagram Walkthrough (High-Level Process Flow)

```mermaid
flowchart TD
    A[User Provides YAML Config] --> B[Configuration Validation]
    B --> C[Trainer Initialization]
    C --> D[GPU Detection & Setup]
    D --> E[Model Creation<br/>YOLO/RTDETR]
    E --> F{MLOps Enabled?}
    F -->|Yes| G[MLflow Setup]
    F -->|No| H[Direct Training]
    G --> I[Hyperparameter<br/>Optimization?]
    H --> I
    I -->|Yes| J[Genetic Algorithm<br/>or ASHA via Ray Tune]
    I -->|No| K[Standard Training]
    J --> L[Training Execution]
    K --> L
    L --> M[Real-time Monitoring]
    M --> N[Artifact Organization]
    N --> O[MLflow Logging]
    O --> P[Model Registration]
    P --> Q[Training Complete]
    
    style A fill:#e1f5fe
    style Q fill:#c8e6c9
    style F fill:#fff3e0
    style I fill:#fff3e0
```

---

## 🗺️ System Workflow (Detailed Sequence Diagram)

```mermaid
sequenceDiagram
    participant User
    participant Config as YAML Config
    participant Trainer as TrainerWrapper
    participant GPU as gpu_utils
    participant MLflow as MLflow Setup
    participant Model as YOLO/RTDETR
    participant Storage as MinIO/S3
    
    User->>Config: Load configuration file
    Config->>Trainer: create_trainer(config_path, trial_number)
    Trainer->>Trainer: load_config()
    Trainer->>Trainer: setup_directories()
    
    Trainer->>GPU: gpu_compatibility_check(force_gpu)
    GPU->>GPU: obtener_info_gpu_json()
    GPU->>GPU: print_gpu_report()
    GPU-->>Trainer: GPU status & config
    
    Trainer->>Model: create_model(model_name, model_type)
    Model-->>Trainer: Model instance
    
    alt MLOps Enabled
        Trainer->>MLflow: set_config_vars()
        MLflow->>MLflow: Configure environment variables
        MLflow->>MLflow: Setup MLflow tracking
        MLflow-->>Trainer: MLflow configured
    end
    
    Trainer->>Model: add_callback("on_train_start")
    Trainer->>Model: add_callback("on_epoch_end")
    Trainer->>Model: add_callback("on_train_end")
    
    User->>Trainer: train(trainer, config, fitness)
    
    alt Hyperparameter Optimization
        Trainer->>Model: tune() - Genetic/ASHA
        Model-->>Trainer: Best parameters
        Trainer->>Trainer: Update config with best params
    end
    
    Trainer->>Model: train(config)
    
    loop Training Epochs
        Model->>Trainer: on_epoch_end()
        Trainer->>MLflow: log_metrics()
        Trainer->>MLflow: log_progress()
        alt Force Stop
            Trainer->>Model: force_stop_train()
        end
    end
    
    Model->>Trainer: on_train_end()
    Trainer->>MLflow: log_artifacts()
    Trainer->>Storage: Store model weights
    Trainer->>MLflow: register_model()
    
    Trainer-->>User: Training complete with results
```

---

## 🏗️ Architecture Components

```mermaid
mindmap
  root((Wyolo Architecture))
    Core Engine
      TrainerWrapper
        Configuration Management
        Model Lifecycle
        Callback Orchestration
      Elemental
        Base Constants
        State Management
        GPU Usage (40% default)
    
    Hardware Layer
      gpu_utils
        GPU Detection
        Compatibility Check
        Resource Management
        Rich Reporting
      GPU Utils Dependencies
        GPUtil
        torch.cuda
        rich
    
    MLOps Integration
      MLflow Setup
        Environment Configuration
        Artifact Organization
        Metadata Collection
        Experiment Tracking
      Storage Layer
        MinIO/S3 Integration
        Local File Management
        Structured Directories
    
    Model Support
      YOLO Models
        Object Detection
        Classification
        Segmentation
      RT-DETR
        Transformer-based
        Real-time Detection
    
    Optimization Engine
      Genetic Algorithm
        Population Evolution
        Parameter Mutation
        Fitness Selection
      ASHA Algorithm
        Ray Tune Integration
        Early Stopping
        Resource Optimization
    
    Utilities
      Configuration
        YAML Parsing
        Validation
        Environment Variables
      Model Wrapper
        MLflowYOLOModel
        PyTorch Integration
        Serialization
    
    External Dependencies
      Ultralytics YOLO
      PyTorch
      MLflow
      Ray Tune
      YAML Processing
```

---

## ⚙️ Container Lifecycle

### Build Process

```mermaid
flowchart TD
    A[Source Code] --> B[pyproject.toml]
    B --> C[Hatchling Build Backend]
    C --> D[Dependency Resolution]
    D --> E[Package Creation]
    E --> F[Wheel Distribution]
    F --> G[Docker Build<br/>Optional]
    G --> H[Container Image]
    
    I[Development] --> J[make install-dev]
    J --> K[Development Dependencies]
    K --> L[Testing & QA Tools]
    
    M[Production] --> N[make install]
    N --> O[Core Dependencies Only]
    O --> P[Optimized Package]
    
    style A fill:#e3f2fd
    style H fill:#e8f5e8
    style L fill:#fff3e0
    style P fill:#fff3e0
```

**Build Steps:**
1. **Source Preparation**: Clean repository with proper structure
2. **Dependency Resolution**: pyproject.toml defines all dependencies
3. **Package Build**: Hatchling creates wheel distribution
4. **Docker Build** (Optional): Container creation for deployment
5. **Quality Assurance**: Testing, linting, type checking via Makefile

### Runtime Process

```mermaid
flowchart TD
    A[Container/Process Start] --> B[Environment Initialization]
    B --> C[Load Environment Variables]
    C --> D[GPU Hardware Detection]
    D --> E[Configuration File Loading]
    E --> F[MLflow Connection Setup]
    F --> G[Directory Structure Creation]
    G --> H[Model Instantiation]
    H --> I[Callback Registration]
    I --> J[Training Loop Start]
    J --> K[Epoch Monitoring]
    K --> L[Artifact Collection]
    L --> M[Progress Logging]
    M --> N[Model Saving]
    N --> O[MLflow Logging]
    O --> P[Cleanup & Shutdown]
    
    style A fill:#e1f5fe
    style P fill:#c8e6c9
    style F fill:#fff3e0
    style J fill:#fff3e0
```

**Runtime Steps:**
1. **Environment Setup**: Load all environment variables and paths
2. **Hardware Detection**: GPU availability and compatibility checks
3. **Configuration Loading**: Parse and validate YAML configuration
4. **MLOps Connection**: Establish MLflow and storage connections
5. **Directory Creation**: Set up organized artifact directories
6. **Model Loading**: Instantiate YOLO/RT-DETR model with weights
7. **Training Execution**: Main training loop with monitoring
8. **Artifact Management**: Organize and store training results
9. **Cleanup**: Proper resource cleanup and shutdown

---

## 📂 File-by-File Guide

| File/Directory | Purpose & Content |
|---------------|-------------------|
| **`src/wyolo/__init__.py`** | Main package entry point exposing `create_trainer` and `train` functions |
| **`src/wyolo/trainer/trainer_wrapper.py`** | Core orchestration engine managing training lifecycle and MLOps integration |
| **`src/wyolo/trainer/utils/mlflow_setup.py`** | MLOps integration handling MLflow configuration, artifact organization, and metadata collection |
| **`src/wyolo/trainer/gpu_utils.py`** | Hardware management for GPU detection, compatibility checking, and resource allocation |
| **`src/wyolo/trainer/cte/elemental.py`** | Base constants and state management for the trainer system |
| **`src/wyolo/trainer/dto/model_wrapper.py`** | MLflow model wrapper for PyTorch model serialization and registration |
| **`datasets_config_examples/`** | Ready-to-use YAML configuration templates for different computer vision tasks |
| **`datasets_config_examples/classification/`** | Classification training configuration with dataset and model parameters |
| **`datasets_config_examples/detection/`** | Object detection training configuration with YOLO-specific settings |
| **`datasets_config_examples/segmentation/`** | Semantic segmentation training configuration with mask annotation support |
| **`pyproject.toml`** | Modern Python package configuration defining dependencies, build system, and metadata |
| **`Makefile`** | Development workflow automation for installation, testing, building, and deployment |
| **`LICENSE`** | MIT license file governing the use and distribution of the software |
| **`README.md`** | Comprehensive project documentation with installation, usage, and architectural guides |

---

---

## ✨ Key Features

### 🎯 **Multi-Task Support**
- **Object Detection**: Train YOLO models for bounding box detection
- **Image Classification**: Support for classification tasks with YOLO-CLS
- **Semantic Segmentation**: Train segmentation models with YOLO-SEG
- **RT-DETR Support**: Real-time DETR transformer models

### 🚀 **Advanced Training Capabilities**
- **Auto-Batch Sizing**: Intelligent batch size optimization based on GPU memory
- **GPU Management**: Automatic GPU detection and resource allocation
- **Mixed Precision Training**: Optimized for modern GPU architectures
- **Early Stopping**: Configurable stopping criteria and callbacks

### 🔧 **MLOps Integration**
- **MLflow Tracking**: Complete experiment logging and versioning
- **MinIO/S3 Storage**: Scalable artifact storage integration
- **DVC Support**: Data version control integration (planned)
- **Model Registry**: Automatic model registration and versioning

### 🧬 **Hyperparameter Optimization**
- **Genetic Algorithms**: Population-based hyperparameter search
- **ASHA Algorithm**: Asynchronous Successive Halving for efficient tuning
- **Ray Tune Integration**: Distributed hyperparameter optimization
- **Custom Search Spaces**: Flexible parameter configuration

### 📊 **Monitoring & Visualization**
- **Real-time Progress**: Live training progress monitoring
- **GPU Metrics**: Hardware utilization tracking
- **Artifact Organization**: Automatic categorization of training outputs
- **Rich Console Output**: Beautiful terminal displays with rich formatting

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)
- MLflow server (for experiment tracking)
- MinIO/S3 storage (for artifact storage)

### Installation

#### Standard Installation
```bash
pip install wyolo
```

#### Development Installation
```bash
git clone https://github.com/wisrovi/wyoloservice2_worker.git
cd wyoloservice2_worker
make install-dev
```

#### Full Installation (with all dependencies)
```bash
make install-all
```

### Quick Start

```python
from wyolo import create_trainer, train

# Create trainer with configuration
trainer, config = create_trainer(
    config_path="examples/detection_config.yaml",
    trial_number=1
)

# Start training
final_result = train(
    trainer=trainer,
    request_config=config,
    fitness="metrics/mAP50(M)"
)

print(f"Training completed with result: {final_result}")
```

### CLI Usage

```bash
# Train a model directly
wyolo-train --config examples/detection_config.yaml

# Check GPU availability
make check-gpu

# Verify MLflow setup
make check-mlflow
```

---

## 📁 File Structure

```
wyolo/
├── src/
│   └── wyolo/
│       ├── __init__.py                 # Main package entry point
│       └── trainer/
│           ├── trainer_wrapper.py      # Core training orchestration
│           ├── gpu_utils.py           # GPU management utilities
│           ├── dto/
│           │   └── model_wrapper.py   # MLflow model wrapper
│           ├── cte/
│           │   └── elemental.py       # Base constants and utilities
│           └── utils/
│               └── mlflow_setup.py    # MLflow integration setup
├── datasets_config_examples/          # Example configurations
│   ├── classification/
│   │   └── config_train.yaml         # Classification config
│   ├── detection/
│   │   └── config_train.yaml         # Detection config
│   └── segmentation/
│       └── config_train.yaml         # Segmentation config
├── pyproject.toml                     # Package configuration
├── Makefile                          # Development utilities
├── LICENSE                           # MIT License
└── README.md                         # This file
```

### Key Directory Functions

- **`src/wyolo/trainer/`**: Core training orchestration and utilities
- **`datasets_config_examples/`**: Ready-to-use configuration templates
- **`pyproject.toml`**: Package metadata, dependencies, and build configuration
- **`Makefile`**: Development workflow automation

---

## ⚙️ Configuration & Setup

### Environment Variables

Configure these environment variables for optimal performance:

```bash
# GPU Usage Control
export MAX_GPU=80  # Maximum GPU utilization percentage (0-100)

# MLflow Configuration
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="your-minio-access-key"
export AWS_SECRET_ACCESS_KEY="your-minio-secret-key"

# Worker Metadata (optional)
export WORKER_HOST="hostname"
export WORKER_OS="linux"
export WORKER_GPU_COUNT="1"
```

### Configuration File Structure

All training configurations use YAML format. Key sections:

#### Basic Configuration
```yaml
model: "yolov8n.pt"        # Pre-trained model to start from
type: "yolo"               # Model type: yolo, rtdetr

train:
  batch: -1                # Auto-batch size (-1 for automatic)
  data: "/path/to/dataset" # Dataset path
  epochs: 100              # Number of training epochs
  imgsz: 640              # Input image size
```

#### MLOps Configuration
```yaml
minio:
  MINIO_ENDPOINT: "http://localhost:9000"
  MINIO_ID: "access-key"
  MINIO_SECRET_KEY: "secret-key"

mlflow:
  MLFLOW_TRACKING_URI: "http://localhost:5000"

sweeper:
  study_name: "my_experiment"
  version: 1
  fitness: "metrics/mAP50(M)"
```

#### Hyperparameter Tuning
```yaml
genetic:
  activate: true
  generations: 10
  poblation_size: 5
  min_epochs_by_ind: 5
  direction: "maximize"
  fitness: "metrics/mAP50(M)"
  use_genetic: false  # Set true for genetic, false for ASHA
```

### Dataset Configuration

#### Classification Dataset
```yaml
train:
  data: "/path/to/classification/dataset"
  # Dataset should follow YOLO classification format:
  # dataset/
  # ├── train/
  # │   ├── class1/
  # │   └── class2/
  # └── val/
  #     ├── class1/
  #     └── class2/
```

#### Detection Dataset
```yaml
train:
  data: "/path/to/dataset/data.yaml"
  # data.yaml should contain:
  # train: /path/to/train/images
  # val: /path/to/val/images
  # nc: number_of_classes
  # names: [class1, class2, ...]
```

#### Segmentation Dataset
```yaml
train:
  data: "/path/to/segmentation/data.yaml"
  # Similar to detection, but with mask annotations
```

---

## 💡 Usage Examples

### Basic Object Detection Training

```python
from wyolo import create_trainer, train

# Load configuration
config_path = "datasets_config_examples/detection/config_train.yaml"

# Create trainer instance
trainer, config = create_trainer(
    config_path=config_path,
    trial_number=1
)

# Execute training
result = train(
    trainer=trainer,
    request_config=config,
    fitness="metrics/mAP50(M)"
)

print(f"Training completed! Final mAP50: {result}")
```

### Image Classification with Genetic Optimization

```python
from wyolo import create_trainer, train

# Configuration with genetic tuning
config_path = "classification_config_with_genetics.yaml"

trainer, config = create_trainer(
    config_path=config_path,
    trial_number=1
)

# Training with automatic hyperparameter optimization
result = train(
    trainer=trainer,
    request_config=config,
    fitness="metrics/accuracy_top1"
)

print(f"Best accuracy achieved: {result}")
```

### GPU-Optimized Training

```python
import os
from wyolo import create_trainer, train

# Set GPU utilization limit
os.environ["MAX_GPU"] = "75"  # Use 75% of GPU memory

trainer, config = create_trainer(
    config_path="gpu_optimized_config.yaml",
    trial_number=1
)

# Force GPU usage (will exit if GPU unavailable)
config["extras"]["force_gpu"] = True

result = train(
    trainer=trainer,
    request_config=config,
    fitness="metrics/mAP50(M)"
)
```

### MLflow Integration Example

```python
from wyolo import create_trainer, train

# Configuration with full MLOps setup
config_path = "mlops_enabled_config.yaml"

trainer, config = create_trainer(
    config_path=config_path,
    trial_number=1
)

# Training with automatic MLflow logging
result = train(
    trainer=trainer,
    request_config=config,
    fitness="metrics/mAP50(B)"
)

# Results are automatically logged to MLflow:
# - Model artifacts
# - Training metrics
# - Hardware information
# - Configuration parameters
```

---

## 📊 System Workflow Diagram

```
┌─────────────────────┐
│   Configuration     │
│   (YAML Config)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────┐
│  Trainer Wrapper    │────▶│  GPU Utils       │
│  - Load Config      │     │  - Check GPU     │
│  - Create Model     │     │  - Set Memory    │
│  - Setup Callbacks  │     └──────────────────┘
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────┐
│  MLflow Setup       │────▶│  Model Training   │
│  - Configure Env    │     │  - YOLO Train    │
│  - Set Tags         │     │  - Callbacks     │
│  - Log Artifacts    │     │  - Monitor       │
└─────────┬───────────┘     └──────────────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────┐
│  Hyperparameter     │────▶│  Results Storage │
│  Optimization       │     │  - MLflow        │
│  - Genetic/ASHA     │     │  - MinIO/S3      │
│  - Ray Tune         │     │  - Local Files   │
└─────────────────────┘     └──────────────────┘
```

---

## 🔄 Diagram Walkthrough

### 1. **Configuration Phase**
- User provides YAML configuration file
- System validates configuration parameters
- Sets up environment variables and paths

### 2. **GPU Detection & Setup**
- `gpu_utils.py` performs hardware detection
- Configures CUDA and GPU memory allocation
- Sets batch size based on available memory

### 3. **Trainer Initialization**
- `TrainerWrapper` creates YOLO/RT-DETR model
- Configures MLflow integration if enabled
- Sets up training callbacks and monitoring

### 4. **Training Execution**
- Executes training with specified parameters
- Real-time monitoring and logging
- Handles early stopping and force termination

### 5. **Hyperparameter Optimization** (Optional)
- Genetic algorithm or ASHA optimization
- Ray Tune integration for distributed tuning
- Automatic best parameter selection

### 6. **Results Management**
- Organizes artifacts into structured directories
- Logs metrics, models, and metadata to MLflow
- Stores large files in MinIO/S3 storage

### 7. **Model Registration**
- Automatically registers trained models
- Version control and metadata tracking
- Ready for deployment pipelines

---

## 🧪 Development & Testing

### Running Tests
```bash
# Run all tests with coverage
make test

# Run fast tests only
make test-fast

# Run specific test file
pytest tests/test_trainer.py -v
```

### Code Quality
```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

# Run all quality checks
make check
```

### Development Setup
```bash
# Complete development environment
make dev-setup

# Start development environment
make start

# Run example
make run-example
```

---

## 🤝 Contributing

We welcome contributions! Please follow our development guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Ensure all quality checks pass (`make check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Use Black for code formatting
- Follow PEP 8 guidelines
- Add type hints for all functions
- Include docstrings for public methods

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Support

- **Documentation**: [https://wyolo.readthedocs.io/](https://wyolo.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/wisrovi/wyoloservice2_worker/issues)
- **Discussions**: [GitHub Discussions](https://github.com/wisrovi/wyoloservice2_worker/discussions)
- **Email**: wisrovi.rodriguez@gmail.com

---

## 🌟 Acknowledgments

- **Ultralytics**: For the amazing YOLO implementation
- **MLflow**: For experiment tracking infrastructure
- **Ray Tune**: For hyperparameter optimization
- **Rich**: For beautiful terminal output

---

## 📈 Roadmap

- [ ] **DVC Integration**: Complete data version control support
- [ ] **Web Dashboard**: Training monitoring interface
- [ ] **Cloud Deployments**: AWS, GCP, Azure deployment templates
- [x] **Model Serving**: Integrated model serving capabilities with LitServe & TensorRT
- [x] **TensorRT Acceleration**: Native FP16 quantization, dynamic batching, and GPU NMS
- [ ] **Multi-GPU Support**: Distributed training across multiple GPUs
- [ ] **Advanced Augmentations**: Enhanced data augmentation pipeline

---

## ⚡ High-Performance TensorRT Inference & Serving (`wyolo.inference`)

Wyolo provides an ultra-fast inference and serving subsystem specifically optimized for NVIDIA GPUs, achieving **5x–10x latency reduction** compared to standard PyTorch inference:

- **Quantization**: Native FP16 half-precision on Tensor Cores.
- **Dynamic Batching**: Dynamic batch sizes (1 to 16 images) with pre-allocated VRAM buffers.
- **Embedded GPU NMS**: Non-Maximum Suppression executed directly inside the TensorRT engine (`EfficientNMS`), outputting canonical detections `[x1, y1, x2, y2, conf, cls]`.
- **Zero-Copy DMA**: Pinned host memory (`cuda.pagelocked_empty`) and asynchronous memory transfers (`memcpy_htod_async`, `memcpy_dtoh_async`).
- **LitServe Integration**: Enterprise-grade HTTP/REST serving with adaptive dynamic batching.

### 1. Exporting YOLO to TensorRT Engine
```python
from wyolo.inference import export_to_engine

# Export PyTorch weights to optimized TensorRT engine
engine_path = export_to_engine(
    model_path="models/yolov8n.pt",
    imgsz=640,
    half=True,       # FP16 quantization
    dynamic=True,    # Dynamic batch size
    batch=16,        # Max batch limit
    nms=True,        # Embedded GPU NMS
    device=0,
)
print(f"Exported engine: {engine_path}")
```

### 2. Python Inference API
```python
from wyolo.inference import YOLOInference

# Initialize unified predictor (supports both .engine and .pt models)
predictor = YOLOInference(
    model_path="models/yolov8n.engine",
    conf=0.25,
    imgsz=640,
)

# Inference on single image (supports file path, numpy array, PIL Image, or raw bytes)
detections = predictor.predict("sample.jpg")

# Direct inference on raw image bytes (ideal for Kafka consumers & network streams)
with open("sample.jpg", "rb") as f:
    detections = predictor.predict_bytes(f.read())

# Batch inference
batch_detections = predictor.predict_batch(["img1.jpg", "img2.jpg"])
```

### 3. Launching LitServe Microservice
```python
from wyolo.inference import serve_yolo

# Launch high-throughput microservice with adaptive dynamic batching
serve_yolo(
    engine_path="models/yolov8n.engine",
    port=8000,
    max_batch_size=16,
    batch_timeout=0.025,  # 25ms aggregation window
    workers_per_device=1,
)
```

---

## 📄 Bibliography & Resources
- [Ultralytics Framework](https://docs.ultralytics.com/)
- [MLflow MLOps Platform](https://mlflow.org/)
- [wpipe Orchestration Library](https://github.com/wisrovi/wpipe)

---

## 👤 Author & Official Affiliation

* **William Steve Rodriguez Villamizar (Wisrovi)**
* **Role:** Principal AI Engineer & Applied AI Solutions Architect | Scientific Researcher
* 📧 **Email:** [wisrovi.rodriguez@gmail.com](mailto:wisrovi.rodriguez@gmail.com)
* 🌐 **Official Portal:** [wisrovi.dev](https://wisrovi.dev)
* 💼 **LinkedIn:** [wisrovi-rodriguez](https://www.linkedin.com/in/wisrovi-rodriguez/)
* 🆔 **ORCID:** [0009-0005-0710-1861](https://orcid.org/0009-0005-0710-1861)
* 📦 **PyPI:** [pypi.org/user/wisrovi/](https://pypi.org/user/wisrovi/)
* 🐙 **GitHub:** [@wisrovi](https://github.com/wisrovi)

---

## 🛠️ Key Technologies and Libraries

- **Python 3.8+**: Modern, typed Python runtime with strict type annotations.
- **Ultralytics YOLO (v8 / v11 / v26)**: SOTA object detection, instance segmentation, pose estimation, and classification engine.
- **NVIDIA TensorRT & PyCUDA**: Hardware-accelerated GPU inference engine with FP16 precision, embedded EfficientNMS, and asynchronous pinned memory DMA transfers.
- **LitServe (Lightning AI)**: High-throughput AI model serving framework featuring adaptive dynamic batching.
- **MLflow**: Experiment tracking, metrics logging, artifact management, and model registry.
- **PyTorch**: Deep learning backend with CUDA acceleration and per-process VRAM allocation ceiling.
- **WPipe & WPipe-Plugins**: State-machine workflow engine powering the post-training analytical pipeline.
- **WRedis**: High-performance corporate Redis client for real-time progress events and metric broadcasts.
- **DVC (Data Version Control)**: Dataset versioning and pipeline reproduction.
- **OpenCV & Albumentations**: Image processing, visual augmentation, and synthetic perturbation analysis.
- **Grad-CAM & Eigen-CAM**: Explainable AI (XAI) feature attribution and visual saliency map generation.
- **Scikit-learn & SciPy**: Statistical validation, AUC insertion/deletion curves, and bootstrap evaluation.
- **Python-docx & PyLaTeX**: Automated generation of DOCX and publication-grade LaTeX reports.
- **Rich & Loguru**: Diagnostic terminal formatting and structured asynchronous logging.
