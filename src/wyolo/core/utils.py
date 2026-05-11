import os
import random
import time
from datetime import datetime
from glob import glob
from typing import List
from PIL import Image
from loguru import logger
from wredis.hash import RedisHashManager


class StatusEDA:
    """Status for EDA operations."""

    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class EDAManager:
    """Manages Exploratory Data Analysis operations."""

    def __init__(self, config: dict):
        self.config = config
        redis_config = config.get("redis", {})
        self.redis_manager = RedisHashManager(
            host=redis_config.get("REDIS_HOST", "localhost"),
            port=redis_config.get("REDIS_PORT", 6379),
            db=redis_config.get("REDIS_DB", 0),
        )

    def save_eda(self, path_results: str):
        """Save EDA results and log to Redis."""
        try:
            self._save_example_images(path_results)
            self._save_dataset_info(path_results)

            # Log to Redis
            self.redis_manager.create_hash(
                "eda",
                f"eda:{self.config.get('task_id', 'unknown')}",
                {
                    "status": StatusEDA.SUCCESS,
                    "path_results": path_results,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error in EDA: {e}")
            self.redis_manager.create_hash(
                "eda",
                f"eda:{self.config.get('task_id', 'unknown')}",
                {
                    "status": StatusEDA.ERROR,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def _save_example_images(self, path_results: str):
        """Save example images from dataset."""
        data_path = self.config.get("train", {}).get("data", "")
        if not data_path or not os.path.exists(data_path):
            return

        # Find images in train folder
        train_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            train_images.extend(glob(os.path.join(data_path, "images", "train", ext)))

        if not train_images:
            return

        # Select random images
        num_examples = min(5, len(train_images))
        example_images = random.sample(train_images, num_examples)

        # Save examples
        example_dir = os.path.join(path_results, "example_images")
        os.makedirs(example_dir, exist_ok=True)

        for i, img_path in enumerate(example_images):
            try:
                img = Image.open(img_path)
                img.save(os.path.join(example_dir, f"example_{i}.jpg"))
            except Exception as e:
                logger.warning(f"Could not save example image {img_path}: {e}")

    def _save_dataset_info(self, path_results: str):
        """Save dataset information."""
        data_path = self.config.get("train", {}).get("data", "")
        if not data_path or not os.path.exists(data_path):
            return

        dataset_info = {
            "data_path": data_path,
            "timestamp": datetime.now().isoformat(),
        }

        # Count images in each split
        for split in ["train", "val", "test"]:
            split_path = os.path.join(data_path, "images", split)
            if os.path.exists(split_path):
                images = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    images.extend(glob(os.path.join(split_path, ext)))
                dataset_info[f"{split}_images"] = len(images)

        # Save dataset info
        with open(os.path.join(path_results, "dataset_info.json"), "w") as f:
            import json

            json.dump(dataset_info, f, indent=2)


class ProgressManager:
    """Manages training progress tracking."""

    def __init__(self, config: dict):
        self.config = config
        redis_config = config.get("redis", {})
        self.redis_manager = RedisHashManager(
            host=redis_config.get("REDIS_HOST", "localhost"),
            port=redis_config.get("REDIS_PORT", 6379),
            db=redis_config.get("REDIS_DB", 0),
        )
        self.task_id = config.get("task_id", "unknown")

    def update_progress(self, epoch: int, metrics: dict):
        """Update training progress."""
        progress_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "status": "training",
            **metrics,
        }

        self.redis_manager.create_hash(
            "progress", f"progress:{self.task_id}", progress_data
        )

    def complete_training(self, final_results: dict):
        """Mark training as completed."""
        completion_data = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "final_results": final_results,
        }

        self.redis_manager.create_hash(
            "progress", f"progress:{self.task_id}", completion_data
        )
