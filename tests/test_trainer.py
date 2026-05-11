import pytest
import yaml
import os
from unittest.mock import MagicMock, patch
from wyolo.core.trainer_wrapper import TrainerWrapper
from wyolo.core.utils import get_datetime

@pytest.fixture
def mock_config(tmp_path):
    config = {
        "model": "yolov8n.pt",
        "type": "detect",
        "train": {
            "data": "coco8.yaml",
            "epochs": 1,
            "imgsz": 320,
            "batch": 4
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return str(config_file), config

def test_get_datetime():
    dt = get_datetime()
    assert len(dt) == 15
    assert "_" in dt

def test_trainer_initialization(mock_config):
    _, config = mock_config
    trainer = TrainerWrapper(config=config)
    assert trainer.config == config
    assert trainer.is_configured is True

@patch("wyolo.core.trainer_wrapper.YOLO")
def test_create_model(mock_yolo, mock_config):
    _, config = mock_config
    trainer = TrainerWrapper(config=config)
    trainer.create_model(model_name="yolov8n.pt", model_type="detect")
    mock_yolo.assert_called_once_with("yolov8n.pt", task="detect")

@patch("wyolo.core.trainer_wrapper.TrainerWrapper.train")
def test_train_call(mock_train, mock_config):
    _, config = mock_config
    trainer = TrainerWrapper(config=config)
    trainer.train(config_train=config["train"])
    mock_train.assert_called_once_with(config_train=config["train"])

def test_config_property(mock_config):
    _, config = mock_config
    trainer = TrainerWrapper(config=config)
    assert trainer.config_train == config
    
    new_config = {"new": "val"}
    trainer.config_train = new_config
    assert trainer.config == new_config
