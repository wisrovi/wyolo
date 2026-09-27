<p align="center">
  <a href="https://pypi.org/project/wyolo/"><img src="https://img.shields.io/pypi/v/wyolo?style=for-the-badge&logo=pypi&color=3b82f6" alt="PyPI version" /></a>
  <a href="https://linkedin.com/in/wisrovi-rodriguez"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  <a href="https://wisrovi.dev"><img src="https://img.shields.io/badge/Author-wisrovi.dev-111827?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Portal" /></a>
  <a href="https://orcid.org/0009-0005-0710-1861"><img src="https://img.shields.io/badge/ORCID-0009--0005--0710--1861-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License" /></a>
</p>

# wyolo: Professional YOLO MLOps Orchestrator

[![Pylint Score](https://img.shields.io/badge/Pylint-9.5%2B-green.svg)](https://www.pylint.org/)
[![Security: Bandit](https://img.shields.io/badge/Security-Bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**wyolo** is an enterprise-grade framework designed to manage the full lifecycle of YOLO and RT-DETR models. By leveraging a robust **State Machine** architecture and native **MLOps** integrations, `wyolo` provides a high-resiliency environment for computer vision training, tracking, and deployment.

---

## 🏗️ Architecture & System Design

### 1. 🚶 Execution Walkthrough (State Machine)
The system operates as a deterministic pipeline governed by conditional logic. It ensures environment integrity before committing expensive GPU resources.

*For a visual flowchart, please refer to [README_GIT.md](README_GIT.md).*

### 2. 🗺️ Detailed System Workflow
Sequence of operations between the orchestrator, specialized states, and external MLOps entities.

*For the sequence interaction diagram, please refer to [README_GIT.md](README_GIT.md).*

### 3. 🗺️ Architecture Components
A layered view of the ecosystem, illustrating the separation between orchestration, business logic, and infrastructure.

*For the architectural mindmap, please refer to [README_GIT.md](README_GIT.md).*

---

## ✨ Key Features & Performance

- **🚀 State-Driven Orchestration:** Complex logic implemented via `wpipe` for maximum resiliency.
- **🛡️ Quality-First Design:** Maintains **Pylint > 9.5** and regular **Bandit** security audits.
- **📊 Native MLOps:** Built-in **MLflow** tracking, automated model registry, and **DVC** data versioning.
- **⚡ Resource Efficiency:** Real-time monitoring of Peak RAM, CPU usage, and VRAM optimization.
- **📦 Distributed Architecture:** Designed to run as an independent worker container (`wtrain-service`).

---

## ⚙️ Lifecycle Management

### a. Build Process (CI/CD)
1.  **Multi-Stage Dockerfile:** Optimizes image size while providing CUDA/CUDNN support.
2.  **Environment Isolation:** Automatic resolution of complex dependencies (RT-DETR, MLflow, Redis).
3.  **Sanity Checks:** Static analysis executed during the build phase.

### b. Runtime Process (Execution)
1.  **Initialization:** `train_service.sh` triggers the service wrapper.
2.  **Discovery:** Automatic discovery of GPU topology and dataset paths.
3.  **Orchestration:** The Pipeline engine manages retries, timeouts, and state transitions.
4.  **Persistence:** Results and logs are persisted to `wtrain.db` (SQLite WAL) for audit trails.

---

## 📂 File-by-File Guide

| Component | Description |
|:---|:---|
| `src/wyolo/app/main.py` | Entry point. Configures the `wpipe` state machine and orchestrates the worker. |
| `src/wyolo/app/states/` | Atomic state implementations (Check GPU, Dataset, MinIO, Train). |
| `src/wyolo/core/` | Core logic for the `TrainerWrapper` and `MLflowManager`. |
| `src/wyolo/docker/` | Production-ready orchestration scripts and requirements. |
| `src/wyolo/trainer/` | Specialized DTOs and implementation of the Elemental design pattern. |
| `Makefile` | The central command center for installation, testing, and deployment. |
| `index.html` | High-impact landing page for project stakeholders. |

---

## 📂 Project Structure
```text
src/wyolo
├── app
│   ├── main.py                <-- Orchestrator
│   └── states                 <-- State Machine Steps
│       ├── check/             <-- Validation Logic
│       ├── train/             <-- Training Execution
│       └── error_process/     <-- Resiliency Handling
├── core
│   ├── trainer_wrapper.py     <-- Engine Wrapper
│   └── mlflow_manager.py      <-- MLOps Tracking
└── docker
    ├── train_service.sh       <-- Production Entrypoint
    └── requirements.txt       <-- Locked Dependencies
```

---

## 🚀 Installation & Usage

```bash
# 1. Setup environment
make install

# 2. Run linting & security
make lint

# 3. Execute training suite
wyolo-train --config_path my_config.yaml
```

---

## 👨‍💻 Author
**William Rodríguez - wisrovi**  
*Technology Evangelist & AI Solutions Architect*  
[LinkedIn Profile](https://es.linkedin.com/in/wisrovi-rodriguez)

---

## 📄 Bibliography & Resources
- [Ultralytics Framework](https://docs.ultralytics.com/)
- [MLflow MLOps Platform](https://mlflow.org/)
- [wpipe Orchestration Library](https://github.com/wisrovi/wpipe)

---

## 👤 Autor & Afiliación Oficial

* **William Steve Rodriguez Villamizar (Wisrovi)**
* **Cargo:** Principal AI Engineer & Applied AI Solutions Architect | Scientific Researcher
* 📧 **Email:** [wisrovi.rodriguez@gmail.com](mailto:wisrovi.rodriguez@gmail.com)
* 🌐 **Portal Oficial:** [wisrovi.dev](https://wisrovi.dev)
* 💼 **LinkedIn:** [wisrovi-rodriguez](https://www.linkedin.com/in/wisrovi-rodriguez/)
* 🆔 **ORCID:** [0009-0005-0710-1861](https://orcid.org/0009-0005-0710-1861)
* 📦 **PyPI:** [pypi.org/user/wisrovi/](https://pypi.org/user/wisrovi/)
* 🐙 **GitHub:** [@wisrovi](https://github.com/wisrovi)
