class Elemental:
    GPU_USE = 0.4  # procentaje de uso de GPU
    
    is_configured = False
    start_time = 0
    end_time = 0
    firts_epoch = True
    model = None