import inspect
"""Evaluator that uses Bootstrapping to compute confidence intervals.

This module provides the BootstrapEvaluator WPipe state, which calculates
the mean and a confidence interval (e.g., 95%) for model predictions
using bootstrap resampling. This is critical for R&D to demonstrate
statistical significance.

Inferencias Estadísticas e Intervalos de Confianza (BootstrapEvaluator)

Paper: Statistical Comparison of Classifiers over Multiple Data Sets
Autores: Janez Demšar (Journal of Machine Learning Research - JMLR 2006)
Por qué es el referente: Es la "biblia" metodológica de la revisión por pares cuando se evalúan
    algoritmos de Machine Learning. Explica rigurosamente cómo aplicar pruebas no paramétricas
    (como la prueba de rangos con signo de Wilcoxon y el test de Friedman) e intervalos de confianza
    mediante Bootstrapping para verificar si las diferencias en métricas son estadísticamente
    significativas ($p < 0.05$) o si son fruto de la variabilidad muestral.
Referencia / Cita: JMLR 7 (2006): 1-30.
"""

import json
import os
import numpy as np
from sklearn.utils import resample
from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="BootstrapEvaluator", version="v1.0")
class BootstrapEvaluator:
    """WPipe step for Bootstrap evaluation."""

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext) -> PostTrainContext:
        """Executes the Bootstrap evaluation.

        Args:
            ctx (PostTrainContext): The pipeline context containing predictions.

        Returns:
            PostTrainContext: The context with bootstrapping results attached.
        """
        # TODO: Get actual predictions and targets from ctx in the future
        predictions: list[float] = []
        targets: list[float] = []
        n_bootstraps: int = 1000
        ci: float = 0.95

        if not predictions or not targets:
            return ctx

        bootstrapped_scores: list[float] = []
        preds_arr = np.array(predictions)
        targets_arr = np.array(targets)

        for _ in range(n_bootstraps):
            indices = resample(np.arange(len(preds_arr)))
            sub_preds = preds_arr[indices]
            sub_targets = targets_arr[indices]

            score = np.mean((sub_preds > 0.5) == (sub_targets > 0.5))
            bootstrapped_scores.append(float(score))

        mean_score = float(np.mean(bootstrapped_scores))
        std_err = float(np.std(bootstrapped_scores))

        lower_p = ((1.0 - ci) / 2.0) * 100
        upper_p = (ci + ((1.0 - ci) / 2.0)) * 100
        ci_lower = float(np.percentile(bootstrapped_scores, lower_p))
        ci_upper = float(np.percentile(bootstrapped_scores, upper_p))

        results = {
            "mean": mean_score,
            "std": std_err,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

        output_dir = os.path.join(ctx.project_path, "extras", "bootstrap")
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

        with open(
            os.path.join(output_dir, "bootstrap_results.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=4)

        return ctx
