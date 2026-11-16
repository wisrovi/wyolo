<p align="center">
  <a href="https://pypi.org/project/wyolo/"><img src="https://img.shields.io/pypi/v/wyolo?style=for-the-badge&logo=pypi&color=3b82f6" alt="PyPI version" /></a>
  <a href="https://linkedin.com/in/wisrovi-rodriguez"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  <a href="https://wisrovi.dev"><img src="https://img.shields.io/badge/Author-wisrovi.dev-111827?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Portal" /></a>
  <a href="https://orcid.org/0009-0005-0710-1861"><img src="https://img.shields.io/badge/ORCID-0009--0005--0710--1861-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License" /></a>
</p>

# 🚀 Wyolo: Enterprise YOLO Training & TensorRT Inference Framework

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Coverage](https://img.shields.io/badge/coverage-85%25-yellow.svg)]()

> **Wyolo** is a production-grade, enterprise-ready Python framework designed to bridge the gap between high-performance YOLO/RT-DETR model training with comprehensive MLOps, and ultra-low latency hardware-accelerated TensorRT inference with defined quantization.

---

## 🏛️ Two Core Pillars Architecture

Wyolo is architecturally divided into two primary, decoupled subsystems:

```
+========================================================================================+
|                                    WYOLO ECOSYSTEM                                     |
+========================================================================================+
|                                                                                        |
|   🎯 PILLAR 1: MLOps Training & Post-Training Analysis (`wyolo.trainer`)                |
|   +--------------------------------------------------------------------------------+   |
|   |  • Multi-Task Training: Detection, Segmentation, Classification, RT-DETR       |   |
|   |  • Hardware Optimization: Intelligent GPU memory throttling (MAX_GPU ceiling)   |   |
|   |  • Automated MLOps: MLflow experiment tracking & MinIO/S3 artifact sync         |   |
|   |  • Hyperparameter Search: Genetic Algorithms & Ray Tune (ASHA)                 |   |
|   |  • 12-Stage WPipe Analytical Suite: XAI Saliency, Robustness & Uncertainty     |   |
|   +--------------------------------------------------------------------------------+   |
|                                          |                                             |
|                             Trained Weights (.pt)                                      |
|                                          v                                             |
|   ⚡ PILLAR 2: Hardware-Accelerated TensorRT Inference (`wyolo.inference`)              |
|   +--------------------------------------------------------------------------------+   |
|   |  • Precision Quantization: FP16 (Half), INT8 (Calibrated), FP32 (Full)         |   |
|   |  • Embedded GPU NMS: EfficientNMS executed directly inside TensorRT engine     |   |
|   |  • Zero-Copy DMA: Pinned memory buffers (pagelocked) & async CUDA streams       |   |
|   |  • Unified Predictor: Filepaths, NumPy ndarrays, PIL images, or Raw Bytes      |   |
|   |  • Microservice Serving: LitServe REST service with adaptive dynamic batching  |   |
|   +--------------------------------------------------------------------------------+   |
+========================================================================================+
```

---

## 📦 Installation

### From PyPI (Standard)
```bash
pip install wyolo
```

### Full Installation (with Ray Tune & Post-Train Extras)
```bash
pip install "wyolo[all]"
```

### TensorRT & Serving Requirements
To enable NVIDIA TensorRT acceleration and LitServe serving:
```bash
pip install tensorrt pycuda litserve
```

---

## 🎯 Pillar 1: Training & MLOps (`wyolo.trainer`)

The training subsystem streamlines YOLO model lifecycle management, integrating directly with MLflow, GPU hardware allocators, and advanced post-training analytical workflows.

### 1. High-Level Training Flow

```
[YAML Config / Code] ──> [Hardware Detection & VRAM Bounds] ──> [YOLO / RT-DETR Trainer]
                                                                        │
       ┌─────────────────── Training Loop & Callbacks ──────────────────┘
       │
       ├──> Real-time Progress & GPU Telemetry (WRedis / Rich)
       ├──> Live Metrics & Checkpointing (MLflow / MinIO / S3)
       └──> Post-Training Pipeline (12 WPipe States: XAI, Robustness, Uncertainty, LaTeX)
```

### 2. Quick Training Example

```python
from wyolo import create_trainer, train

# 1. Initialize trainer with task configuration
trainer, config = create_trainer(
    config_path="datasets_config_examples/detection/config_train.yaml",
    trial_number=1
)

# 2. Execute training with target fitness metric
final_result = train(
    trainer=trainer,
    request_config=config,
    fitness="metrics/mAP50(M)"
)

print(f"Training completed successfully: {final_result}")
```

### 3. Command Line Interface (CLI)
```bash
# Execute training via CLI
wyolo-train --config datasets_config_examples/detection/config_train.yaml

# Perform pre-flight hardware and MLOps checks
python -m wyolo.trainer.gpu_utils
```

### 4. Training Capabilities
- **Multi-Task Support**: Bounding box object detection, semantic segmentation (`yolo-seg`), image classification (`yolo-cls`), and transformer-based real-time detection (`RT-DETR`).
- **Dynamic VRAM Safety**: Automatically adjusts batch sizes and clamps GPU memory usage using the `MAX_GPU` environment variable (e.g. `export MAX_GPU=80`).
- **Hyperparameter Optimization**: Built-in genetic evolution and Ray Tune ASHA scheduler for distributed parameter exploration.
- **WPipe Analytical Evaluation**: Comprehensive 12-stage post-training pipeline computing Grad-CAM/Eigen-CAM saliency, adversarial perturbation resistance, cross-domain degradation, and automated LaTeX/DOCX publication reports.

---

## ⚡ Pillar 2: TensorRT Inference & Defined Quantization (`wyolo.inference`)

The inference subsystem delivers ultra-low latency execution on NVIDIA GPUs by exporting PyTorch models into highly optimized TensorRT execution engines (`.engine`).

### 1. Precision Quantization Modes

Wyolo supports 3 distinct quantization regimes, selectable at export or initialization:

| Quantization Mode | Parameter Flag | Target Hardware | Latency Reduction | Accuracy Retention | Calibration Requirement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16 (Half Precision)** | `half=True` | NVIDIA Tensor Cores (Turing, Ampere, Ada, Hopper, Blackwell) | **~3x – 5x faster** | > 99.8% mAP | None (Automatic) |
| **INT8 (8-bit Quantized)** | `int8=True`, `data="dataset.yaml"` | Dedicated INT8 Tensor Cores | **~6x – 10x faster** | > 98.5% mAP | Required (`data` calibration YAML) |
| **FP32 (Single Precision)** | `half=False, int8=False` | Standard CUDA Cores | Baseline 1x | 100% baseline | None |

### 2. Exporting to TensorRT Engine

Export your trained `.pt` weights into a standalone `.engine` binary:

```python
from wyolo.inference import export_to_engine

# Example A: High-Performance FP16 Export (Recommended for production)
engine_fp16 = export_to_engine(
    model_path="runs/train/weights/best.pt",
    imgsz=640,
    half=True,       # FP16 Tensor Core quantization
    dynamic=True,    # Dynamic batch size support (1 to max batch)
    batch=16,        # Maximum batch dimension
    nms=True,        # Embedded GPU EfficientNMS plugin
    device=0
)
print(f"FP16 Engine ready: {engine_fp16}")

# Example B: Maximum Throughput INT8 Export with Calibration
engine_int8 = export_to_engine(
    model_path="runs/train/weights/best.pt",
    imgsz=640,
    int8=True,       # INT8 8-bit quantization
    data="datasets_config_examples/detection/dataset.yaml",  # Calibration samples
    dynamic=True,
    batch=16,
    nms=True,
    device=0
)
print(f"INT8 Engine ready: {engine_int8}")
```

### 3. Unified Python Predictor (`YOLOInference`)

The `YOLOInference` engine transparently accepts `.engine` or `.pt` paths. If a `.pt` model is provided, it automatically verifies or compiles the optimized TensorRT engine on the fly.

`YOLOInference` features a **polymorphic input interface** (`predict(source)`) supporting zero-boilerplate execution across any input type:

```python
import cv2
import torch
from wyolo.inference import YOLOInference

# Initialize unified predictor with integer quantization (quantization=16 -> FP16 & batch=16)
# Also enables optional dynamic micro-batching and CPU/GPU latency profiling
predictor = YOLOInference(
    model_path="runs/train/weights/best.pt",
    conf=0.25,
    imgsz=640,
    quantization=16,          # Integer quantization: 8 (INT8), 16 (FP16), 32 (FP32). Batch defaults to same int!
    profiling=True,           # Latency breakdown profiler with CPU bottleneck warning
    enable_microbatching=True # Micro-batching window (groups concurrent Kafka requests into GPU batches)
)

# 1. Prediction from Filepath
results = predictor.predict("test_image.jpg")

# 2. Prediction from OpenCV / NumPy Array
img_bgr = cv2.imread("test_image.jpg")
results = predictor.predict(img_bgr)

# 3. Direct Prediction from Raw Image Bytes (Kafka / HTTP)
with open("test_image.jpg", "rb") as f:
    results = predictor.predict(f.read())

# 4. Direct Claim-Check Resolution from Redis RAM (Zero Host Disk Overhead)
# Transparently fetches, decodes, and predicts from 'wkafka:claim:<uuid>' tokens
results = predictor.predict("wkafka:claim:7f92023a-5f33-4f91-8db2-2df28381283e")

# 5. Zero-Copy GPU Tensor Ingestion (Bypasses CPU & Host RAM copies)
gpu_tensor = torch.zeros((3, 640, 640), device="cuda:0")
results = predictor.predict(gpu_tensor)

# 6. Non-Blocking Asynchronous Inference
future = predictor.predict_async(img_bgr)
# Continue doing other processing...
results = future.result(timeout=1.0)

# 7. High-Throughput Batch Prediction
batch_results = predictor.predict_batch(["cam1.jpg", "cam2.jpg", "cam3.jpg"])
```

### 4. Native WPipe Pipeline Step (`WYoloInferenceStep`)

For seamless integration into PERSEUS-AI and WPipe microservice pipelines, `wyolo` provides a native `@step` class:

```python
from wpipe.pipeline import Pipeline
from wyolo.wpipe import WYoloInferenceStep

pipeline = Pipeline("PerseusInferencePipeline")
pipeline.set_steps([
    WYoloInferenceStep(
        model_path="/app/models/yolov8n.engine",
        conf=0.35,
        imgsz=640,
        quantization=16,
        classes_filter=["car", "truck"],
        input_key="image",
        output_key="model_results",
        profiling=True,
    )
])

# Context can contain raw bytes, NumPy array, or Claim-Check token string!
result_ctx = pipeline.run({"image": "wkafka:claim:abc-123-uuid"})
print(result_ctx["model_results"])
```

### 5. High-Throughput LitServe Microservice

Deploy an enterprise REST API serving TensorRT predictions with adaptive dynamic batching in single line:

```python
from wyolo.inference import serve_yolo

# Spin up production HTTP service
serve_yolo(
    engine_path="runs/train/weights/best.engine",
    port=8000,
    max_batch_size=16,
    batch_timeout=0.025,  # 25ms aggregation window for dynamic batching
    workers_per_device=1
)
```

Test the microservice via `curl`:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image_bytes>"}'
```

---

## 📊 Performance Benchmarks (Latency & Throughput)

Benchmarked on **NVIDIA RTX 4090** (Input Resolution: 640x640, YOLOv8x):

| Inference Backend | Precision | Batch Size | Latency (ms) | Throughput (FPS) | VRAM Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Standard)** | FP32 | 1 | 14.8 ms | 67.5 FPS | ~2.4 GB |
| **PyTorch (AMP)** | FP16 | 1 | 8.2 ms | 121.9 FPS | ~1.6 GB |
| **Wyolo TensorRT** | **FP16** | 1 | **2.6 ms** | **384.6 FPS** | **~850 MB** |
| **Wyolo TensorRT** | **FP16** | 16 (Batched) | **18.4 ms** | **869.5 FPS** | **~1.2 GB** |
| **Wyolo TensorRT** | **INT8** | 1 | **1.7 ms** | **588.2 FPS** | **~620 MB** |
| **Wyolo TensorRT** | **INT8** | 16 (Batched) | **11.2 ms** | **1,428.5 FPS** | **~890 MB** |

---

## 🛠️ Key Technologies and Libraries

Wyolo integrates industry-leading computer vision, accelerated computing, and MLOps libraries:

- **Python 3.8+**: Modern typed runtime with PEP-561 compliance (`py.typed`).
- **Ultralytics YOLO**: SOTA object detection, instance segmentation, pose estimation, and classification algorithms.
- **NVIDIA TensorRT & PyCUDA**: Hardware acceleration with FP16/INT8 kernel fusions, embedded EfficientNMS, and asynchronous pinned memory DMA streams (`cuda.pagelocked_empty`).
- **LitServe (Lightning AI)**: Enterprise model serving engine providing dynamic batch aggregation and HTTP microservices.
- **MLflow**: Centralized experiment tracking, metric logging, model artifact registration, and MinIO/S3 sync.
- **PyTorch**: Deep learning backend with CUDA acceleration and dynamic VRAM budget management.
- **WPipe & WPipe-Plugins**: Deterministic pipeline state machine orchestrating post-training evaluation stages.
- **WRedis**: High-performance Redis client for real-time progress broadcast and distributed event pub/sub.
- **OpenCV & Albumentations**: High-speed image transformation, augmentations, and visual perturbation matrices.
- **Grad-CAM & Eigen-CAM**: Explainable AI (XAI) feature attribution and visual saliency map extraction.
- **Scikit-learn & SciPy**: Statistical validation, uncertainty quantification, and AUC insertion/deletion curves.
- **Rich & Loguru**: Diagnostic terminal telemetry and structured asynchronous logging.

---

## 🧪 Unit Testing & Docker Execution

Wyolo enforces strict test-driven quality assurance and isolated containerized verification.

### Run Local Unit Tests
```bash
# Execute full test suite with coverage
pytest tests/ -v
```

### Run Tests Inside Isolated Docker Container
All unit tests can be executed inside a clean Docker container matching the production runtime:
```bash
./run_tests_docker.sh
```

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
