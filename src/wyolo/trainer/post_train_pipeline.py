from wpipe import Pipeline
from wpipe.exception.api_error import ProcessError

# ImageECamYOLO hace uso de Grad-CAM (https://github.com/xaviercanche/Grad-CAM) para explicar el modelo, en que se fija y por qué toma decisiones, generando mapas de calor sobre las imágenes de entrada. Esto ayuda a entender el comportamiento del modelo y a identificar posibles sesgos o errores en la predicción.
# Basado en: https://arxiv.org/pdf/1610.02391.pdf, https://arxiv.org/pdf/1610.02391v1.pdf
from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO

from .states import (
    CleanFolderExtra,
    LlmAnalyzer,
    PostTrain,
    BootstrapEvaluator,
    LatexExporter,
    ModelComplexityProfiler,
    OutlierFailureAnalyzer,
    RobustnessNoiseEvaluator,
    AdversarialAttackTester,
    CrossDomainGeneralizer,
    FeatureRepresentationAnalyzer,
    QuantitativeXAIValidator,
    UncertaintyQuantifier,
    ModelFocusDescriber
)
from .safe_step import SafeStep, safe_step

db_path = "output/tracking.db"  # Path to tracking database for to save metrics, events, alerts and execution history (with error capture)
config_dir = "configs"

pipeline_post_train = Pipeline(
    pipeline_name="professional_post_train_pipeline",
    pipeline_version="1.0.0",
    verbose=False,  # Toggle detailed logging for debugging and monitoring
    show_progress=False,  # Display a progress bar during pipeline execution
)

pipeline_post_train.set_steps(
    [
        CleanFolderExtra(),
        PostTrain(),
        safe_step(
            ImageECamYOLO(
                ECAMConfig(
                    confidence_threshold=0.10,
                )
            )
        ),
        safe_step(ModelFocusDescriber()),
        safe_step(QuantitativeXAIValidator()),
        safe_step(BootstrapEvaluator()),
        safe_step(ModelComplexityProfiler()),
        safe_step(OutlierFailureAnalyzer()),
        safe_step(RobustnessNoiseEvaluator()),
        safe_step(AdversarialAttackTester()),
        safe_step(CrossDomainGeneralizer()),
        safe_step(FeatureRepresentationAnalyzer()),
        safe_step(UncertaintyQuantifier()),
        safe_step(LatexExporter()),
        safe_step(LlmAnalyzer()),
    ]
)


if __name__ == "__main__":
    initial_data_dict = (
        ...
    )  # define your initial data dictionary here for the pipeline run

    try:
        result = pipeline_post_train.run(initial_data_dict)
    except ProcessError as e:
        print(f"Error occurred: {e}")
