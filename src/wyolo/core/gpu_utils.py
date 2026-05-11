import json
import GPUtil
from ultralytics.utils.autobatch import autobatch


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
                f"gpu_{gpu.id}_name": gpu.name,
                f"gpu_{gpu.id}_uuid": gpu.uuid,
                f"gpu_{gpu.id}_memoryTotal": gpu.memoryTotal,
                f"gpu_{gpu.id}_memoryFree": gpu.memoryFree,
                f"gpu_{gpu.id}_memoryUsed": gpu.memoryUsed,
                f"gpu_{gpu.id}_load": gpu.load * 100,
                f"gpu_{gpu.id}_temperature": gpu.temperature,
            }

            if hasattr(gpu, "processes"):
                gpu_data["processes"] = [
                    {
                        "pid": process.pid,
                        "name": process.name,
                        "memoryUsed": getattr(
                            process, "memoryUsed", getattr(process, "memoryUsage", 0)
                        ),
                    }
                    for process in gpu.processes
                ]
            else:
                gpu_data["processes"] = "Processes information not available."

            gpu_info.append(gpu_data)

        return json.dumps(gpu_info, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error al obtener información de GPU: {str(e)}"})


def get_better_batch(trainer, batch_to_use: int = 32) -> int:
    """
    Calcula el tamaño óptimo de batch para la GPU disponible.
    """
    try:
        optimal_batch = autobatch(
            model=trainer.model,
            imgsz=trainer.config.get("train", {}).get("imgsz", 640),
            fraction=trainer.GPU_USE,
            batch_size=batch_to_use,
        )
        return optimal_batch
    except Exception as e:
        print(f"Error calculating optimal batch size: {e}")
        return batch_to_use
