import os
import boto3
from botocore.client import Config
from loguru import logger
from wpipe import step, to_obj
from states.utils.util import get_base_config

@step(name="check_minio_buckets", version="v1.0", tags=["check_minio"])
@to_obj
def check_minio_buckets(data_input):
    """Asegura que los buckets necesarios existan en MinIO."""
    
    # Obtenemos la configuración base que tiene las credenciales y el endpoint corregido
    from states.utils.util import read_base_config
    cfg = read_base_config()
    
    minio_cfg = cfg.get("minio", {})
    endpoint = minio_cfg.get("MINIO_ENDPOINT")
    access_key = minio_cfg.get("MINIO_ID")
    secret_key = minio_cfg.get("MINIO_SECRET_KEY")
    
    # Buckets requeridos por el sistema
    required_buckets = ["models", "mlflow-artifacts", "dvcstorage"]
    
    try:
        s3 = boto3.resource(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1' # MinIO suele usar esta por defecto
        )
        
        for bucket_name in required_buckets:
            bucket = s3.Bucket(bucket_name)
            if bucket.creation_date:
                logger.info(f"✅ Bucket '{bucket_name}' ya existe.")
            else:
                logger.info(f"🔨 Creando bucket '{bucket_name}'...")
                bucket.create()
                logger.info(f"✅ Bucket '{bucket_name}' creado exitosamente.")
                
    except Exception as e:
        logger.error(f"❌ Error al verificar/crear buckets en MinIO: {e}")
        # No lanzamos excepción para no detener el pipeline si el usuario ya los creó manualmente
        # pero dejamos el log del error.
        
    return {"minio_status": 1}
