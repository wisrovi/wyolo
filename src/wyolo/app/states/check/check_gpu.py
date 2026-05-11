import os
import subprocess
import torch
from loguru import logger
from wpipe import step, to_obj

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
)


@step(name="check_gpu_available", version="v1.0", tags=["check_gpu_available"])
@to_obj
def check_gpu_available(data_input):
    test_command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ]
    print("-" * 50)
    print(f"DEBUG: CUDA Runtime Version: {torch.version.cuda}")
    print(
        f"DEBUG: Driver Version: {subprocess.check_output(test_command).decode().strip()}"
    )

    # Hito: Intento de despertar el driver a nivel sistema
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except:
        pass

    if not torch.cuda.is_available():
        # Aquí podrías usar tus "prints bonitos" con colorama o rich
        print("⚠️ GPU no detectada. Reintentando...")
        raise RuntimeError("La GPU no respondió a tiempo.")

    return {"gpu_status": int(True)}
