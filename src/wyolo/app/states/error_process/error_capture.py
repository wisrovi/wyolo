from loguru import logger
from wpipe import to_obj
from wpipe.decorators import step

ERROR_LOG_FOLDER = "/results/errors"


@step(name="error_capture", version="v1.0", tags=["error_capture"])
@to_obj
def error_capture(context, error):

    print("\n" + "!" * 60)
    print("🚨 ALERTA DE SISTEMA: ERROR DETECTADO")
    print("!" * 60)
    print(f"📍 ESTADO FALLIDO: {error['step_name']}")
    print(f"📄 ARCHIVO: {error['file_path']}")
    print(f"🔢 LÍNEA: {error['line_number']}")
    print(f"⚠️ MENSAJE: {error['error_message']}")
    print(f"⚠️ MENSAJE: {error['error_traceback']}")
    print("-" * 60)
    # print("🔍 INFO DE LA BODEGA (CONTEXTO):")
    # print(f"   context: {context} ")
    # print("-" * 60)

    return context
