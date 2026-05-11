# Getting Started

## Installation

```bash
pip install wyolo
```

## Basic Usage

```python
from wyolo.core.trainer_wrapper import TrainerWrapper

config = {
    "model": "yolov8n.pt",
    "type": "detect",
    "train": {
        "data": "coco128.yaml",
        "epochs": 1,
        "imgsz": 320
    }
}

trainer = TrainerWrapper(config=config)
trainer.create_model(model_name="yolov8n.pt", model_type="detect")
trainer.train(config_train=config["train"])
```
