from wpipe import PipelineContext


class PostTrainContext(PipelineContext):
    """Shared context for the post-training pipeline.

    Attributes:
        model_path: Path to the trained YOLO model used for post-training predictions.
        project_path: Absolute path where artifacts and results are stored.
        images_test_path: Path to the dataset YAML (used to locate test images).
    """

    model_path: str
    project_path: str
    images_test_path: str
    output_dir: str = ""
    image_data: list = []
