from states.utils.util import get_complete_config, get_user_config
from loguru import logger
from wpipe import step, to_obj
import os
from pydantic import BaseModel


class UserInput(BaseModel):
    user_config_train: str


@step(name="load_yaml", version="v1.0", tags=["load_yaml"])
@to_obj(UserInput)
def load_yaml(data_input: UserInput):
    if not os.path.exists(data_input.user_config_train):
        raise FileNotFoundError(
            f"El archivo de configuración no existe: {data_input.user_config_train}"
        )

    _, USER_CONFIG = get_user_config(user_config_file=data_input.user_config_train)

    return {"user_config": USER_CONFIG}
