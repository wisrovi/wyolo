import os
from glob import glob

import yaml
from ultralytics import YOLO
from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="PostTrain", version="v1.0")
class PostTrain:
    MAX_IMAGES_TO_PROCESS = 10  # Limit to processing up to 10 images for now

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext):
        model_path = ctx.model_path
        images_test_path = ctx.images_test_path
        project_path = ctx.project_path

        try:
            model = YOLO(model_path) if model_path else None
        except Exception as e:
            print(f"[PostTrain] Error loading YOLO model: {e}")
            model = None

        all_images = self._find_images(images_test_path)

        print(
            f"[PostTrain] Using {len(all_images)} images for post-training predictions."
        )

        images_used_for_prediction = []

        images_to_process = all_images[: self.MAX_IMAGES_TO_PROCESS]
        ctx.image_data = images_to_process

        for image in images_to_process:
            try:
                if not hasattr(model, "predict"):
                    raise AttributeError("The model does not have a 'predict' method.")

                model.predict(
                    image,
                    save=True,
                    conf=0.15,
                    exist_ok=True,
                    project=os.path.join(project_path, "extras"),
                    name="post_train_results",
                    verbose=False,
                )

                images_used_for_prediction.append(image)
            except Exception as e:
                print(f"[PostTrain] Error processing image {image}: {e}")

        print(f"[PostTrain] Done. Predictions saved to {ctx.output_dir}.")
        return {
            "image_data": images_used_for_prediction or all_images[: self.MAX_IMAGES_TO_PROCESS],
            "model_path": model_path,
            "images_test_path": images_test_path,
            "project_path": project_path,
            "output_dir": ctx.output_dir,
        }

    def _find_images(self, images_test_path: str) -> list[str]:
        """Locate images for prediction: test, then val/valid, then train as fallback.

        The data path may be a detection dataset YAML (absolute image dirs are read
        from its content, since the config points to a temp copy of the YAML) or a
        classification directory.
        """
        image_dirs = self._resolve_image_dirs(images_test_path)

        if image_dirs:
            candidates = []
            for k in ("test", "val", "valid", "train"):
                if k in image_dirs:
                    candidates.append(os.path.join(image_dirs[k], "*"))
        else:
            folder_path = (
                os.path.dirname(images_test_path)
                if os.path.isfile(images_test_path)
                else images_test_path
            )
            candidates = [
                os.path.join(folder_path, "test", "images", "*"),
                os.path.join(folder_path, "test", "*", "*"),
                os.path.join(folder_path, "val", "images", "*"),
                os.path.join(folder_path, "val", "*", "*"),
                os.path.join(folder_path, "valid", "images", "*"),
                os.path.join(folder_path, "valid", "*", "*"),
            ]

        for pattern in candidates:
            images = [
                img
                for img in glob(pattern)
                if os.path.isfile(img) and self._is_valid_image(img)
            ]
            if images:
                return images

        train_images = []
        if image_dirs:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                train_images.extend(glob(os.path.join(image_dirs["train"], ext)))
        else:
            folder_path = (
                os.path.dirname(images_test_path)
                if os.path.isfile(images_test_path)
                else images_test_path
            )
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                train_images.extend(
                    glob(os.path.join(folder_path, "train", "images", ext))
                )
                train_images.extend(glob(os.path.join(folder_path, "train", "*", ext)))
        if train_images:
            print("[PostTrain] No test/val images found, falling back to train images.")
            return [
                img
                for img in train_images
                if os.path.isfile(img) and self._is_valid_image(img)
            ]

        return []

    def _resolve_image_dirs(self, images_test_path: str) -> dict:
        """Read absolute image dirs (train/val/test) from a detection dataset YAML.

        Returns an empty dict when the path is a directory (classification dataset)
        or the YAML cannot be read.
        """
        if not images_test_path or not os.path.isfile(images_test_path):
            return {}

        try:
            with open(images_test_path, "r") as file:
                data_yaml_config = yaml.safe_load(file) or {}

            dirs = {}
            yaml_dir = os.path.dirname(images_test_path)
            root_path = data_yaml_config.get("path", "")
            if root_path:
                if not os.path.isabs(root_path):
                    root_path = os.path.normpath(os.path.join(yaml_dir, root_path))
            else:
                root_path = yaml_dir

            for split in ("train", "val", "test"):
                split_path = data_yaml_config.get(split)
                if not isinstance(split_path, str):
                    continue

                # Check absolute
                abs_split = (
                    split_path
                    if os.path.isabs(split_path)
                    else os.path.normpath(os.path.join(root_path, split_path))
                )

                if os.path.isdir(abs_split):
                    dirs[split] = abs_split
                elif os.path.isdir(os.path.join(abs_split, "images")):
                    dirs[split] = os.path.join(abs_split, "images")
            return dirs
        except Exception as e:
            print(f"[PostTrain] Could not read data yaml {images_test_path}: {e}")
            return {}

    @staticmethod
    def _is_valid_image(img: str) -> bool:
        """Filter out YOLO metric plots and charts from candidate images."""
        basename_lower = os.path.basename(img).lower()
        path_lower = img.lower()

        if any(folder in path_lower for folder in ("runs/", "post_train_results/")):
            return False
        if "confusion_matrix" in basename_lower or "curve" in basename_lower:
            return False
        if basename_lower.startswith(("train_batch", "val_batch", "results")):
            return False
        if basename_lower in ("labels.jpg", "labels_correlogram.jpg"):
            return False
        return True
