"""GPU utility functions and diagnostic reporting for wyolo training."""

import json
import os
import sys

import GPUtil
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def obtener_info_gpu_json():
    """Obtain detailed GPU diagnostics and metrics as a list of dictionaries.

    Returns:
        list: List of GPU metric dictionaries or error objects.
    """
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return [{"error": "No se encontraron GPUs disponibles."}]

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
        return [{"error": f"Ocurrió un error al obtener la información de la GPU: {e}"}]


def print_gpu_report(hardware_gpu_count):
    """Print formatted GPU diagnostic summary to console.

    Args:
        hardware_gpu_count: Number of hardware GPUs detected.
    """
    console = Console()

    # 1. Hito: Encabezado de Éxito
    console.print(
        Panel.fit(
            "✅ [bold green]ÉXITO: GPU detectada y lista para entrenamiento[/bold green]",
            border_style="green",
            title="[bold]Hardware Status[/bold]",
        )
    )

    # 2. Hito: Tabla de Detalles Técnicos
    device_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)

    table = Table(
        show_header=True,
        header_style="bold cyan",
        title="\n[bold]Detalles de la GPU[/bold]",
    )
    table.add_column("Parámetro", style="dim")
    table.add_column("Valor", justify="right")

    table.add_row("Modelo GPU", f"[bold white]{props.name}[/bold white]")
    table.add_row("Capacidad Computacional", f"{props.major}.{props.minor}")
    table.add_row("Streaming Multiprocessors (SMs)", str(props.multi_processor_count))
    table.add_row("Memoria Total", f"{props.total_memory / 1024**2:.0f} MB")

    console.print(table)

    # 3. Hito: Entorno y Software
    # Usamos un Panel pequeño para las variables de entorno
    env_info = (
        f"[bold blue]CUDA Version:[/bold blue] {torch.version.cuda}\n"
        f"[bold blue]cuDNN Enabled:[/bold blue] {'[green]Yes[/green]' if torch.backends.cudnn.enabled else '[red]No[/red]'}\n"
        f"[bold blue]GPU Count:[/bold blue] {hardware_gpu_count}\n"
        f"[bold blue]CUDA_VISIBLE_DEVICES:[/bold blue] [yellow]{os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}[/yellow]"
    )

    console.print(
        Panel(
            env_info,
            title="[bold]Software Environment[/bold]",
            border_style="blue",
            expand=False,
        )
    )

    # 4. Hito: Recursos del Contenedor
    res_table = Table(
        show_header=True,
        header_style="bold magenta",
        title="\n[bold]Recursos Asignados (Límites Docker)[/bold]",
    )
    res_table.add_column("Recurso", style="dim")
    res_table.add_column("Asignación", justify="right")

    max_gpu_env = os.environ.get("MAX_GPU", "-1")
    gpu_asignada = f"{max_gpu_env}%" if max_gpu_env != "-1" else "Sin límite (100%)"
    res_table.add_row("Uso máximo VRAM (Autobatch)", gpu_asignada)

    limit_ram = os.environ.get("WORKER_RAM_MEMORY")
    if limit_ram:
        ram_total = f"{limit_ram}"
    else:
        try:
            import psutil

            ram_total = f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
        except Exception:
            ram_total = "Desconocido"
    res_table.add_row("RAM Total Disponible", ram_total)

    limit_cpu = os.environ.get("WORKER_CPU_CORES")
    if limit_cpu:
        cpu_count = f"{limit_cpu}"
    else:
        try:
            cpu_count = str(len(os.sched_getaffinity(0)))
        except AttributeError:
            cpu_count = str(os.cpu_count())
    res_table.add_row("CPU Cores (Afinidad)", cpu_count)

    limit_shm = os.environ.get("WORKER_SHM_MEMORY") or os.environ.get(
        "WORKER_RAM_MEMORY"
    )
    if limit_shm:
        shm_size = f"{limit_shm}"
    else:
        try:
            shm_stats = os.statvfs("/dev/shm")
            shm_size = f"{(shm_stats.f_bsize * shm_stats.f_blocks) / 1024**3:.1f} GB"
        except Exception:
            shm_size = "Desconocido"
    res_table.add_row("Memoria Compartida (SHM)", shm_size)

    console.print(res_table)


def gpu_compatibility_check(force_gpu: bool):
    """Check GPU hardware availability and CUDA compatibility.

    Args:
        force_gpu (bool): Whether training execution strictly requires GPU.

    Returns:
        bool: True if compatible GPU is available, False otherwise.
    """
    available_gpu = torch.cuda.is_available()
    hardware_gpu_count = torch.cuda.device_count()

    there_is_gpu = available_gpu and hardware_gpu_count > 0

    if force_gpu:
        print("Checking GPU availability for training...")
        print("Have been forced to use GPU.")

        if not available_gpu:
            if hardware_gpu_count > 0:
                print("GPU hardware detected but CUDA is not available.")
                error = ""

                try:
                    torch.cuda.init()
                except Exception as e:
                    error = str(e)

                print(f"Driver python error: {error}")
                sys.exit(1)
            else:
                print("No GPU hardware detected.")
                sys.exit(1)

        if hardware_gpu_count < 1:
            print("No GPU hardware detected.")
            sys.exit(1)

    if there_is_gpu:
        print_gpu_report(hardware_gpu_count)
        return True
    else:
        print("Forcing CPU usage for training.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        torch.cuda.is_available = lambda: False

        return False
