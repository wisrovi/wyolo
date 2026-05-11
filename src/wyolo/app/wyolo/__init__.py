"""Wyolo - Professional YOLO Training Library with MLOps integration."""

from .trainer.trainer_wrapper import create_trainer, train

__version__ = "1.0.0"
__author__ = "William Steve Rodriguez Villamizar"
__email__ = "wisrovi.rodriguez@gmail.com"

__all__ = ["train", "create_trainer"]