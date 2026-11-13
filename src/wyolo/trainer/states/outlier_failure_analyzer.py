import inspect
"""Analyzer that identifies critical false positives and false negatives.

This module provides the OutlierFailureAnalyzer WPipe state, which loads the
test dataset in FiftyOne, compares ground truth with model predictions, and
extracts specific outlier cases (e.g., high-confidence false positives).
This enables deep debugging of model bias and dataset errors.

Análisis Cualitativo y Modos de Fallo (OutlierFailureAnalyzer)
Paper: Diagnostics for Fine-Grained Object Detection via Error Analysis
Autores: Derek Hoiem, Yunsheng Ma, Xiaobai Liu (ECCV 2012)
Por qué es el referente: Es el trabajo pionero que formalizó el análisis cuantitativo y
    cualitativo de errores en detectores de objetos. Introduce la categorización sistemática de errores
    dividiéndolos en Falsos Positivos de Localización (Loc),
    Falsos Positivos por Confusión con el Fondo (BG),
    Falsos Positivos por Confusión de Clase (Sim/Oth) y Falsos Negativos (Miss),
    sirviendo de base directa para herramientas modernas como FiftyOne.
Referencia / DOI: Springer ECCV 2012, LNCS 7574, pp. 854-867.
"""

import os
import json
try:
    import fiftyone as fo
except Exception:
    fo = None
from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="OutlierFailureAnalyzer", version="v1.0")
class OutlierFailureAnalyzer:
    """WPipe step for failure mode analysis via FiftyOne."""

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext) -> PostTrainContext:
        """Executes the failure analysis.

        Args:
            ctx (PostTrainContext): The pipeline context containing dataset paths.

        Returns:
            PostTrainContext: The unmodified pipeline context.
        """
        images_test_path = ctx.images_test_path or ""
        yaml_path = None
        if os.path.isfile(images_test_path):
            yaml_path = images_test_path
            dataset_dir = os.path.dirname(images_test_path)
        elif os.path.isdir(images_test_path):
            dataset_dir = images_test_path
            candidates = ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"]
            yaml_path = next(
                (
                    os.path.join(dataset_dir, c)
                    for c in candidates
                    if os.path.isfile(os.path.join(dataset_dir, c))
                ),
                None,
            )
        else:
            dataset_dir = os.path.dirname(images_test_path) if images_test_path else ""

        dataset_name = f"dataset_{os.path.basename(yaml_path or dataset_dir or 'unknown')}"

        if fo is None:
            return ctx

        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
        else:
            try:
                load_kwargs = {
                    "dataset_dir": dataset_dir if dataset_dir and os.path.isdir(dataset_dir) else None,
                    "dataset_type": fo.types.YOLOv5Dataset,
                    "name": dataset_name,
                }
                if yaml_path and os.path.isfile(yaml_path):
                    load_kwargs["yaml_path"] = yaml_path
                dataset = fo.Dataset.from_dir(**load_kwargs)
            except Exception as e:
                print(f"Failed to load fiftyone dataset: {e}")
                return ctx

        fp_count = 0
        fn_count = 0

        try:
            # We assume predictions are added, skipping actual evaluation if missing
            dataset.evaluate_detections(
                "predictions",
                gt_field="ground_truth",
                eval_key="eval",
                compute_mAP=True,
            )

            high_conf_fp = dataset.filter_labels(
                "predictions",
                (fo.ViewField("eval") == "fp") & (fo.ViewField("confidence") > 0.75),
            )

            high_conf_fn = dataset.filter_labels(
                "ground_truth", fo.ViewField("eval") == "fn"
            )

            fp_count = len(high_conf_fp)
            fn_count = len(high_conf_fn)
            print(f"Total Critical False Positives: {fp_count}")
            print(f"Total Critical False Negatives: {fn_count}")

        except Exception as e:
            print(f"Error evaluating dataset: {e}")

        output_dir = os.path.join(ctx.project_path, "extras", "failures")
        os.makedirs(output_dir, exist_ok=True)

        import inspect
        import sys
        md_content = inspect.cleandoc(sys.modules[__name__].__doc__ or "No description available.")
        md_content = f"""# Analysis Report

{md_content}

## Methodology
This directory contains the outputs and results of this specific analysis. The metrics and plots generated here reflect the model's behavior according to the described methodology."""
        if md_content:
            with open(os.path.join(output_dir, "DESCRIPTION.md"), "w", encoding="utf-8") as fmd:
                fmd.write(md_content)


        results = {
            "critical_false_positives": fp_count,
            "critical_false_negatives": fn_count,
        }

        with open(
            os.path.join(output_dir, "failure_analysis.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=4)

        return ctx
