"""Native WPipe step for unified YOLO inference in microservices."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-few-public-methods

from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from wpipe.steps import step
except ImportError:
    # Fallback dummy decorator if wpipe is not installed in standard build environment
    def step(name: Optional[str] = None):  # type: ignore
        """Fallback step decorator when wpipe is not installed.

        Args:
            name (Optional[str]): Step name.

        Returns:
            Callable: Decorator function.
        """

        def decorator(cls: Any) -> Any:
            """Decorate class with step name.

            Args:
                cls (Any): Target class.

            Returns:
                Any: Decorated class.
            """
            cls.__step_name__ = name or cls.__name__
            return cls

        return decorator


from wyolo.inference.predictor import YOLOInference


@step(name="WYoloInferenceStep")
class WYoloInferenceStep:
    """Standardized corporate WPipe step executing high-performance YOLO object detection."""

    model_path: str = ""
    conf: float = 0.25
    imgsz: int = 640
    quantization: int = 16
    classes_filter: Optional[List[str]] = None
    input_key: str = "image"
    output_key: str = "model_results"
    profiling: bool = False

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        quantization: Optional[int] = None,
        classes_filter: Optional[List[str]] = None,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        profiling: Optional[bool] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        """Initialize WYoloInferenceStep.

        Args:
            model_path (Optional[str]): Path to model file. Defaults to class attribute.
            conf (Optional[float]): Confidence threshold. Defaults to class attribute.
            imgsz (Optional[int]): Square image dimension. Defaults to class attribute.
            quantization (Optional[int]): Quantization integer (8, 16, 32). Defaults to 16.
            classes_filter (Optional[List[str]]): Target classes to filter. Defaults to None.
            input_key (Optional[str]): Context key containing image or claim.
            output_key (Optional[str]): Context key to store detections.
            profiling (Optional[bool]): Enable latency profiling. Defaults to False.
            redis_client (Optional[Any]): Redis client for Claim-Check resolution. Defaults to None.
        """
        self.model_path = model_path or self.model_path
        self.conf = conf if conf is not None else self.conf
        self.imgsz = imgsz or self.imgsz
        self.quantization = quantization or self.quantization
        self.classes_filter = classes_filter or self.classes_filter
        self.input_key = input_key or self.input_key
        self.output_key = output_key or self.output_key
        self.profiling = profiling if profiling is not None else self.profiling

        self.predictor: Optional[YOLOInference] = None
        if self.model_path:
            self.predictor = YOLOInference(
                model_path=self.model_path,
                conf=self.conf,
                imgsz=self.imgsz,
                quantization=self.quantization,
                redis_client=redis_client,
                profiling=self.profiling,
            )

    def __call__(self, context: Any) -> Any:
        """Execute WPipe step pipeline callback.

        Args:
            context (Any): Pipeline context (dict or object).

        Returns:
            Any: Updated pipeline context with detection results under output_key.
        """
        if self.predictor is None:
            if not self.model_path:
                logger.error("WYoloInferenceStep: No model_path configured.")
                return context
            self.predictor = YOLOInference(
                model_path=self.model_path,
                conf=self.conf,
                imgsz=self.imgsz,
                quantization=self.quantization,
                profiling=self.profiling,
            )

        # Extract image source from context without truthiness ambiguity on NumPy arrays
        source: Any = None
        if isinstance(context, dict):
            if self.input_key in context and context[self.input_key] is not None:
                source = context[self.input_key]
            elif "value" in context and context["value"] is not None:
                source = context["value"]
        elif (
            hasattr(context, self.input_key)
            and getattr(context, self.input_key) is not None
        ):
            source = getattr(context, self.input_key)
        elif hasattr(context, "value") and getattr(context, "value") is not None:
            source = getattr(context, "value")

        if source is None:
            logger.warning(
                f"WYoloInferenceStep: Input key '{self.input_key}' not found in context."
            )
            return context

        detections: List[Dict[str, Any]] = self.predictor.predict(source)

        if self.classes_filter:
            detections = [
                d
                for d in detections
                if d.get("name") in self.classes_filter
                or d.get("classname") in self.classes_filter
            ]

        if isinstance(context, dict):
            context[self.output_key] = detections
        else:
            setattr(context, self.output_key, detections)

        return context
