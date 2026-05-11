import subprocess

from loguru import logger
from wpipe import to_obj, step
import os

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    data: dict


class Dataset(BaseModel):
    train: DatasetConfig
    val: DatasetConfig
    test: DatasetConfig


class UserInput(BaseModel):
    user_config: dict


DATASET_FOLDER = "/wyolo/control_server/datasets/"


@step(name="check_dataset", version="v1.0", tags=["check_dataset"])
@to_obj(UserInput)
def check_dataset(input_data: UserInput):
    if not os.path.exists(DATASET_FOLDER):
        logger.error(f"Dataset folder '{DATASET_FOLDER}' does not exist.")
        raise FileNotFoundError(f"Dataset folder '{DATASET_FOLDER}' does not exist.")

    # list of files in dataset folder
    files = os.listdir(DATASET_FOLDER)
    if len(files) == 0:
        # mount the folder
        COMMAND = "sh /usr/local/bin/mount-cifs.sh"
        subprocess.run(COMMAND, shell=True, check=True)

    DATASET = input_data.user_config.train.data
    DATASET = DATASET.replace("/datasets/", DATASET_FOLDER)

    if not os.path.exists(DATASET):
        logger.error(f"Dataset path '{DATASET}' does not exist.")
        raise FileNotFoundError(f"Dataset path '{DATASET}' does not exist.")

    logger.info(f"Dataset path '{DATASET}' exists and is ready for training.")

    return {"dataset_status": int(True)}
