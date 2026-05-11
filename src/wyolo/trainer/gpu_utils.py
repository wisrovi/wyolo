import json

import GPUtil


def obtener_info_gpu_json():
    """
    Obtiene información detallada sobre las GPUs disponibles y la devuelve en formato JSON.
    Maneja el caso en que la propiedad 'processes' no esté disponible.
    """
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return json.dumps({"error": "No se encontraron GPUs disponibles."})

        gpu_info = []
        for gpu in gpus:
            gpu_data = {
                # "gpu_id": gpu.id,
                f"gpu_{gpu.id}_name": gpu.name,
                f"gpu_{gpu.id}_uuid": gpu.uuid,
                f"gpu_{gpu.id}_memoryTotal": gpu.memoryTotal,
                f"gpu_{gpu.id}_memoryFree": gpu.memoryFree,
                f"gpu_{gpu.id}_memoryUsed": gpu.memoryUsed,
                f"gpu_{gpu.id}_load": gpu.load * 100,
                f"gpu_{gpu.id}_temperature": gpu.temperature,
            }
            # Verifica si la propiedad 'processes' existe antes de intentar acceder a ella.
            if hasattr(gpu, "processes"):
                gpu_data["processes"] = [
                    {
                        "pid": process.pid,
                        "name": process.name,
                        "memory": process.memoryUsed,
                    }
                    for process in gpu.processes
                ]
            else:
                gpu_data["processes"] = "Processes information not available."

            gpu_info.append(gpu_data)

        return gpu_info

    except Exception as e:
        return {"error": f"Ocurrió un error al obtener la información de la GPU: {e}"}
