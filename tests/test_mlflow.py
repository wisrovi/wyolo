import pytest
from unittest.mock import MagicMock, patch
from wyolo.core.mlflow_manager import MLFlowManager

def test_mlflow_manager_initialization():
    config = {
        "mlflow": {
            "uri": "http://localhost:5000",
            "experiment_name": "test_exp"
        }
    }
    with patch("mlflow.set_tracking_uri") as mock_uri, \
         patch("mlflow.set_experiment") as mock_exp:
        manager = MLFlowManager(config=config)
        mock_uri.assert_called_once_with("http://localhost:5000")
        mock_exp.assert_called_once_with("test_exp")

@patch("mlflow.start_run")
def test_start_run(mock_start_run):
    config = {"mlflow": {"uri": "test", "experiment_name": "test"}}
    manager = MLFlowManager(config=config)
    manager.start_run(run_name="test_run")
    mock_start_run.assert_called_once_with(run_name="test_run")

@patch("mlflow.log_param")
def test_log_params(mock_log_param):
    config = {"mlflow": {"uri": "test", "experiment_name": "test"}}
    manager = MLFlowManager(config=config)
    params = {"lr": 0.01, "epochs": 10}
    manager.log_params(params)
    assert mock_log_param.call_count == 2
