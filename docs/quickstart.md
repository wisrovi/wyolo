# Quick Start

Get up and running with `wyolo` in minutes.

## Basic Training

```python
from wyolo.core.trainer_wrapper import TrainerWrapper

# Define your configuration
config = {
    "model": "yolov8n.pt",
    "type": "detect",
    "train": {
        "data": "coco128.yaml",
        "epochs": 5,
        "imgsz": 640
    }
}

# Initialize the trainer
trainer = TrainerWrapper(config=config)
trainer.create_model(model_name="yolov8n.pt", model_type="detect")

# Start training
trainer.train(config_train=config["train"])
```

## CLI Usage

The library installs a `wyolo-train` script:

```bash
wyolo-train --config_path my_config.yaml
```
