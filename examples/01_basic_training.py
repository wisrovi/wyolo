"""
Basic YOLOv8 training example using wyolo.
This script demonstrates how to initialize the trainer and start a simple training loop.
"""

from wyolo.core.trainer_wrapper import TrainerWrapper

def main():
    # 1. Define minimal configuration
    config = {
        "model": "yolov8n.pt",  # Pre-trained model
        "type": "detect",       # Task type (detect, segment, classify, pose)
        "train": {
            "data": "coco8.yaml", # Small dataset for testing
            "epochs": 3,          # Number of epochs
            "imgsz": 640,         # Image size
            "batch": 8,           # Batch size
            "project": "examples_output",
            "name": "basic_exp"
        }
    }

    # 2. Initialize the TrainerWrapper
    # It automatically handles environment setup and configuration validation
    trainer = TrainerWrapper(config=config)

    # 3. Create/Load the model
    print(f"--- Creating model: {config['model']} ---")
    trainer.create_model(
        model_name=config["model"],
        model_type=config["type"]
    )

    # 4. Execute Training
    print("--- Starting Training ---")
    results = trainer.train(config_train=config["train"])

    if results:
        print("--- Training Completed Successfully ---")
        print(f"Results saved in: {results.save_dir}")
    else:
        print("--- Training Failed ---")

if __name__ == "__main__":
    main()
