# mypy: ignore-errors
# pylint: disable=all
# ruff: noqa

import os
from pathlib import Path
import subprocess
import csv

class TrainingReportAnalyzer:
    """
    Generate AI-assisted training analysis using OpenCode with fallback.
    """

    OPENCODE_BIN = "/root/.opencode/bin/opencode"
    DEFAULT_MODEL = "opencode/nemotron-3.5-lightning-free"

    @classmethod
    def _get_model(cls) -> str:
        return os.getenv("LLM_MODEL", cls.DEFAULT_MODEL)

    def analyze(
        self,
        results_file: str | Path
    ) -> str:
        """
        Generate a professional training report.

        Args:
            results_file: Path to YOLO results.csv file.

        Returns:
            Generated report text.
        """

        results_file = Path(results_file)

        if not results_file.exists():
            raise FileNotFoundError(
                f"Results file not found: {results_file}"
            )

        # Try OpenCode first with timeout
        try:
            report = self._analyze_with_opencode(results_file)
            if report and len(report.strip()) > 50:
                return report
        except Exception as exc:
            print(f"OpenCode analysis failed, using fallback: {exc}")

        # Fallback: generate basic report from CSV data
        try:
            return self._generate_fallback_report(results_file)
        except Exception as exc:
            print(f"Fallback report generation failed: {exc}")
            return (
                "TRAINING SUMMARY\n"
                "Training completed, but no detailed report could be generated.\n"
                "\n"
                "METRICS ANALYSIS\n"
                "Review the evaluation_metrics results.csv for detailed metrics.\n"
                "\n"
                "CONCLUSION\n"
                "Training run finished successfully; detailed analysis unavailable."
            )

    def _analyze_with_opencode(self, results_file: Path) -> str | None:
        """Attempt to generate report using OpenCode with timeout."""
        prompt = """
        Generate a professional technical training report in English.

        Use correct grammar, punctuation, and spelling.

        Do not use markdown.

        Analyze the training that was performed.

        Evaluate:
        - Training progression.
        - Convergence.
        - Metrics obtained.
        - Possible overfitting.
        - Possible underfitting.
        - Overall model quality.
        - Detected risks.
        - Recommendations.

        Use the following sections:

        TRAINING SUMMARY

        METRICS ANALYSIS

        CONCLUSION

        Maximum three lines per section.

        Do not invent data.
        """

        model = self._get_model()

        result = subprocess.run(
            [
                self.OPENCODE_BIN,
                "run",
                "--model",
                model,
                prompt,
                "-f",
                str(results_file)
            ],
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"OpenCode error:\n{result.stderr}"
            )

        output = result.stdout.strip()
        if not output or len(output) < 50:
            raise RuntimeError("OpenCode returned empty or too short output")

        return output

    @staticmethod
    def _as_float(value, default: float = 0.0) -> float:
        """Safely convert a CSV cell to float, returning default on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _generate_fallback_report(self, results_file: Path) -> str:
        """Generate a basic report from CSV data when OpenCode fails."""
        try:
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as exc:
            return f"Error reading results file: {exc}"

        if not rows:
            return "No training data available for analysis."

        # Parse key metrics (handle different column names)
        epochs = len(rows)
        last_row = rows[-1]
        first_row = rows[0]

        def col(row, *names: str) -> str:
            for name in names:
                if name in row and row[name] not in (None, ""):
                    return row[name]
            return "N/A"

        train_loss = col(last_row, 'train/loss', 'train_loss', 'train/box_loss', 'train/total_loss')
        val_loss = col(last_row, 'val/loss', 'val_loss', 'val/box_loss', 'val/total_loss')
        precision = col(last_row, 'metrics/precision', 'precision', 'metrics/precision(B)')
        recall = col(last_row, 'metrics/recall', 'recall', 'metrics/recall(B)')
        mAP50 = col(last_row, 'metrics/mAP50(B)', 'mAP50', 'metrics/mAP50')
        mAP50_95 = col(last_row, 'metrics/mAP50-95(B)', 'mAP50-95', 'metrics/mAP50-95')
        accuracy = col(last_row, 'metrics/accuracy_top1', 'accuracy', 'metrics/accuracy')

        # Check for overfitting
        train_loss_first = col(first_row, 'train/loss', 'train_loss', 'train/box_loss', 'train/total_loss')
        train_loss_last = train_loss
        val_loss_first = col(first_row, 'val/loss', 'val_loss', 'val/box_loss', 'val/total_loss')
        val_loss_last = val_loss

        overfitting_risk = "Low"
        try:
            tl_first, tl_last = float(train_loss_first), float(train_loss_last)
            vl_first, vl_last = float(val_loss_first), float(val_loss_last)
            if tl_last < tl_first and vl_last > vl_first:
                overfitting_risk = "High"
            elif vl_last > vl_first * 1.2:
                overfitting_risk = "Medium"
        except (ValueError, TypeError):
            pass

        train_trend = "stable/increasing"
        val_trend = "increasing/stable"
        try:
            train_trend = (
                "decreasing" if float(train_loss_last) < float(train_loss_first)
                else "stable/increasing"
            )
            val_trend = (
                "decreasing" if float(val_loss_last) < float(val_loss_first)
                else "increasing/stable"
            )
        except (ValueError, TypeError):
            pass

        # Build report
        report_lines = [
            "TRAINING SUMMARY",
            f"Training completed over {epochs} epochs. Final training loss: {train_loss}, validation loss: {val_loss}.",
            f"Model achieved accuracy: {accuracy}, mAP@50: {mAP50}, mAP@50-95: {mAP50_95}.",
            "",
            "METRICS ANALYSIS",
            f"Precision: {precision}, Recall: {recall}. mAP@50: {mAP50}, mAP@50-95: {mAP50_95}.",
            f"Overfitting risk: {overfitting_risk}. Training loss trend: {train_trend}.",
            f"Validation loss trend: {val_trend}.",
            "",
            "CONCLUSION",
            f"Model trained for {epochs} epochs with final accuracy {accuracy}.",
            f"Overfitting risk assessed as {overfitting_risk}. Consider regularization or early stopping if risk is high.",
            "Review confusion matrix and per-class metrics for detailed performance breakdown."
        ]

        return "\n".join(report_lines)
