# debug:
# from lib.src.wyolo.trainer.trainer_wrapper import create_trainer, train

# production:
from pydantic import BaseModel
from mlflow.store.db.base_sql_model import Base
from wyolo.trainer.trainer_wrapper import create_trainer, train

import os
import tempfile
from loguru import logger
from wpipe import step, to_obj
import yaml

from states.utils.util import get_complete_config, get_user_config
from ..check.check_dataset import DATASET_FOLDER


class UserInput(BaseModel):
    user_config_train: str


@step(name="train_model", version="v1.0", tags=["train_model"])
@to_obj(UserInput)
def train_model(data_input: UserInput):

    if not os.path.exists(data_input.user_config_train):
        raise FileNotFoundError(
            f"El archivo de configuración no existe: {data_input.user_config_train}"
        )

    user_config_path, USER_CONFIG = get_user_config(
        user_config_file=data_input.user_config_train
    )

    config_dict, config_path = get_complete_config(user_config=user_config_path)

    with tempfile.TemporaryDirectory(prefix="wyolo_") as temp_dir:
        config_dict["tempfile"] = temp_dir

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        trainer, request_config = create_trainer(
            config_path=config_path, trial_number=1
        )

        if "train" in request_config:
            DATASET = request_config.get("train").get("data")
            DATASET = DATASET.replace("/datasets/", DATASET_FOLDER)

            request_config["train"]["data"] = DATASET

            fitness = config_dict.get("sweeper", {}).get("fitness", "fitness")
            request_config = train(trainer, request_config, fitness)

        print(request_config)

    return request_config
