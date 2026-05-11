import os
import mlflow
from copy import deepcopy
import yaml
import sys


def get_user_config(
    user_config_file: str = "/wyolo/control_server/datasets/clasification/colorball.v8i.multiclass/config_train.yaml",
):
    # Verificamos si el archivo existe antes de intentar abrirlo
    if not os.path.exists(user_config_file):
        print(f"Error: El archivo '{user_config_file}' no existe.")
        sys.exit(1)

    try:
        with open(user_config_file, "r", encoding="utf-8") as archivo:
            # Cargamos el contenido del YAML
            user_config_train_data = yaml.safe_load(archivo)
    except yaml.YAMLError as exc:
        print(f"Error al leer el archivo YAML: {exc}")
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        sys.exit(1)

    os.makedirs("/wyolo/worker/train_service_results/training_artifacts", exist_ok=True)

    with open(
        "/wyolo/worker/train_service_results/training_artifacts/user_config_train.yaml",
        "w",
        encoding="utf-8",
    ) as file:
        yaml.dump(
            user_config_train_data,
            file,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    return user_config_file, user_config_train_data


def read_user_config(task_data: dict):
    if not isinstance(task_data, dict):
        raise Exception("Fail with user config")

    config_path = task_data["config_path"]

    # Leer el archivo YAML del usuario
    config_path = task_data["config_path"]
    try:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)  # Convertir YAML a dict

            # 🚨 Eliminar `defaults` si existe
            user_config.pop("defaults", None)

    except Exception as e:
        raise Exception(f"❌ Error al cargar YAML ({config_path}): {e}")

    return user_config


# Función para actualizar y completar final_config
def merge_configs(default_config, user_config):
    """
    Fusiona dos configuraciones: user_config tiene prioridad sobre default_config.
    Los campos faltantes en user_config se completan con los valores de default_config.

    Args:
        default_config (dict): Configuración predeterminada.
        user_config (dict): Configuración proporcionada por el usuario.

    Returns:
        dict: Configuración final fusionada.
    """
    # Crear una copia profunda de default_config para evitar modificaciones inesperadas

    final_config = deepcopy(default_config)

    # Iterar sobre las claves de user_config y actualizar final_config
    for key, value in user_config.items():
        if (
            isinstance(value, dict)
            and key in final_config
            and isinstance(final_config[key], dict)
        ):
            # Si ambas son diccionarios, fusionar recursivamente
            final_config[key] = merge_configs(final_config[key], value)
        else:
            # Sobrescribir el valor con el proporcionado por el usuario
            final_config[key] = deepcopy(value)

    return final_config


def get_base_config():
    DEFAULT_CONFIG = {}

    CONTROL_HOST = os.getenv("CONTROL_HOST", "localhost")

    # Cargar configuración base
    with open("/app/config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Reemplazar localhost con CONTROL_HOST
    if "mlflow" in cfg and "MLFLOW_TRACKING_URI" in cfg["mlflow"]:
        cfg["mlflow"]["MLFLOW_TRACKING_URI"] = cfg["mlflow"][
            "MLFLOW_TRACKING_URI"
        ].replace("localhost", CONTROL_HOST)
    if "minio" in cfg and "MINIO_ENDPOINT" in cfg["minio"]:
        cfg["minio"]["MINIO_ENDPOINT"] = cfg["minio"]["MINIO_ENDPOINT"].replace(
            "localhost", CONTROL_HOST
        )
    if "redis" in cfg and "REDIS_HOST" in cfg["redis"]:
        cfg["redis"]["REDIS_HOST"] = cfg["redis"]["REDIS_HOST"].replace(
            "localhost", CONTROL_HOST
        )

    # Solo configurar MLflow si existe en la configuración
    if "mlflow" in cfg:
        mlflow.set_tracking_uri(cfg["mlflow"]["MLFLOW_TRACKING_URI"])
        os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["MLFLOW_TRACKING_URI"]
        # Enable system metrics collection
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    DEFAULT_CONFIG.update(cfg)

    # Solo configurar MinIO si existe en la configuración
    if "minio" in cfg:
        minio_id = os.getenv("CIFS_USER", cfg["minio"].get("MINIO_ID", "mlflow"))
        minio_key = os.getenv("CIFS_PASS", cfg["minio"].get("MINIO_SECRET_KEY", "wyoloservice"))
        
        DEFAULT_CONFIG["minio"]["MINIO_ID"] = minio_id
        DEFAULT_CONFIG["minio"]["MINIO_SECRET_KEY"] = minio_key
        
        # Configurar variables de entorno para que MLflow pueda subir artifacts a MinIO/S3
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = DEFAULT_CONFIG["minio"]["MINIO_ENDPOINT"]
        os.environ["AWS_ACCESS_KEY_ID"] = minio_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = minio_key
        # Opcional: Desactivar verificación SSL si es necesario
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

    # Solo configurar DVC si existe en la configuración
    if "dvc" in cfg:
        DEFAULT_CONFIG["dvc"]["MINIO_ID"] = os.getenv("CIFS_USER", "mlflow")
        DEFAULT_CONFIG["dvc"]["MINIO_SECRET_KEY"] = os.getenv(
            "CIFS_PASS", "wyoloservice"
        )

    with open("/config/final_config.yaml", "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)


def read_base_config():
    get_base_config()
    base_config = read_user_config(
        task_data={"config_path": "/config/final_config.yaml"}
    )

    return base_config


def get_complete_config(user_config: str):
    os.makedirs("/config", exist_ok=True)

    user_config = read_user_config({"config_path": user_config})
    base_config = read_base_config()

    final_config = merge_configs(
        default_config=base_config,
        user_config=user_config,
    )

    final_config["config_path"] = user_config
    final_config["task_id"] = base_config["task_id"]

    tempfile = final_config["tempfile"]
    os.makedirs(tempfile, exist_ok=True)
    config_path = f"{tempfile}/{final_config['task_id']}.yaml"

    try:
        final_config["config_path"] = config_path

        with open(config_path, "w") as archivo:
            yaml.dump(final_config, archivo, default_flow_style=False)

    except Exception as e:
        raise Exception(f"❌ Error al guardar YAML ({config_path}): {e}")

    return final_config, config_path
