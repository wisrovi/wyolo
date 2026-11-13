import os
import glob
import subprocess
from pathlib import Path

from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext
from ..utils.training_report_analyzer import TrainingReportAnalyzer

from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

@step(name="llm_analyzer", version="v1.0")
class LlmAnalyzer:
    RESULTS_RELATIVE = "evaluation_metrics/results.csv"
    LLM_MD_NAME = "extras/llm/LLM_Report.md"
    LLM_DOCX_NAME = "extras/llm/LLM_Report.docx"
    OPENCODE_BIN = "/root/.opencode/bin/opencode"
    DEFAULT_MODEL = "opencode/nemotron-3.5-lightning-free"

    @classmethod
    def _get_model(cls) -> str:
        return os.getenv("LLM_MODEL", cls.DEFAULT_MODEL)

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext):
        project_path = ctx.project_path

        results_file = os.path.join(project_path, self.RESULTS_RELATIVE)
        llm_md_path = os.path.join(project_path, self.LLM_MD_NAME)
        llm_docx_path = os.path.join(project_path, self.LLM_DOCX_NAME)

        os.makedirs(os.path.dirname(llm_md_path), exist_ok=True)

        # 1. Generate explanation for each research state JSON
        self._explain_research_states(project_path)

        # 2. Main report
        try:
            report = TrainingReportAnalyzer().analyze(results_file)

            # Save MD
            with open(llm_md_path, "w", encoding="utf-8") as f:
                f.write(report)

            # Save DOCX
            try:
                doc = Document()
                media_dir = Path("/app/media")
                wtrain_img = media_dir / "wtrain.jpg"
                wpipe_img = media_dir / "wpipe.jpg"
                logo_img = media_dir / "logo.jpg"

                if wtrain_img.exists():
                    doc.add_picture(str(wtrain_img), width=Inches(6.0))
                    doc.add_paragraph(
                        "WTrain: Sistema completo de entrenamiento distribuido para modelos de Inteligencia Artificial."
                    )
                    doc.add_page_break()

                if wpipe_img.exists():
                    doc.add_picture(str(wpipe_img), width=Inches(6.0))
                    doc.add_paragraph(
                        "El sistema hace uso de WPipe: un framework de pipelines profesional, rápido, eficiente y con características forenses avanzadas."
                    )
                    doc.add_page_break()

                for line in report.split("\n"):
                    if line.startswith("# "):
                        doc.add_heading(line[2:], level=0)
                    elif line.startswith("## "):
                        doc.add_heading(line[3:], level=1)
                    elif line.startswith("### "):
                        doc.add_heading(line[4:], level=2)
                    elif line.strip():
                        doc.add_paragraph(line)

                if logo_img.exists():
                    doc.add_page_break()
                    p = doc.add_paragraph()
                    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                    run = p.add_run()
                    run.add_picture(str(logo_img), width=Inches(3.0))
                    doc.add_paragraph(
                        "WTrain hace parte del paquete de la Wisrovi Suit, desarrollada por William Rodriguez (Wisrovi)."
                    ).alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

                doc.add_page_break()
                p = doc.add_paragraph()
                p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                doc.add_paragraph(
                    "Monitoring with https://docs.ultralytics.com/es/guides/model-monitoring-and-maintenance#documentation"
                ).alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                doc.add_paragraph(
                    "Training with https://docs.ultralytics.com/es/guides/kfold-cross-validation#train-yolo-using-k-fold-data-splits"
                ).alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

                doc.save(llm_docx_path)
            except Exception as e:
                print(f"[LLMAnalyzer] Failed to generate DOCX: {e}")

            print(
                f"[LLMAnalyzer] Report written to {llm_md_path} and {llm_docx_path} "
                f"({len(report)} chars)."
            )
            return {
                "llm_report": report,
                "llm_md_path": llm_md_path,
                "llm_docx_path": llm_docx_path,
            }
        except Exception as exc:
            print(f"[LLMAnalyzer] Failed: {exc}")
            return {"llm_report": "", "llm_md_path": "", "error": str(exc)}

    def _explain_research_states(self, project_path: str):
        """Finds JSON files in extras/ and uses OpenCode to explain their contents collectively."""
        extras_dir = os.path.join(project_path, "extras")
        if not os.path.exists(extras_dir):
            return
            
        json_files = glob.glob(os.path.join(extras_dir, "*", "*.json"))
        if not json_files:
            return
            
        analysis_md_path = os.path.join(extras_dir, "GLOBAL_RESEARCH_EXPLANATION.md")
        
        prompt = """
        Eres un experto investigador en inteligencia artificial. Te proporciono múltiples archivos JSON 
        con los resultados de diferentes análisis forenses y validaciones (XAI, ruido, ataques adversarios, complejidad, etc.) 
        de un modelo YOLO en la carpeta 'extras/'.
        Analiza todos estos datos en conjunto y escribe un informe ejecutivo detallado (GLOBAL RESEARCH EXPLANATION) 
        que cruce la información de los diferentes módulos. Explica qué significan estos resultados numéricos, 
        qué aportan al entendimiento general del modelo y para qué sirven en un entorno de investigación. 
        Sé directo, profesional y redacta en Markdown usando títulos, listas y negritas. No superes los 5 párrafos.
        """
        
        model = self._get_model()
        cmd = [self.OPENCODE_BIN, "run", "--model", model, prompt]
        for jf in json_files:
            cmd.extend(["-f", jf])
            
        print(f"[LLMAnalyzer] Generating global research explanation for {len(json_files)} JSON files")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8"
            )
            
            if result.returncode == 0 and len(result.stdout.strip()) > 50:
                with open(analysis_md_path, "w", encoding="utf-8") as fmd:
                    fmd.write(f"# Global Research Analysis Report\n\n{result.stdout.strip()}")
                print(f"[LLMAnalyzer] Global report saved to {analysis_md_path}")
            else:
                self._write_fallback_research(extras_dir, json_files)
        except Exception as e:
            print(f"[LLMAnalyzer] Error generating global explanation: {e}")
            self._write_fallback_research(extras_dir, json_files)

    def _write_fallback_research(self, extras_dir: str, json_files: list):
        """Write a minimal fallback research explanation when LLM is unavailable."""
        analysis_md_path = os.path.join(extras_dir, "GLOBAL_RESEARCH_EXPLANATION.md")
        names = [Path(jf).stem for jf in json_files]
        fallback = (
            "# Global Research Analysis Report\n\n"
            "The LLM service was unavailable to generate a detailed cross-module analysis.\n\n"
            "## Modules Analyzed\n\n"
            + "\n".join(f"- **{n}**" for n in names)
            + "\n\nRefer to the individual JSON files in the extras/ subdirectories for raw metrics.\n"
        )
        with open(analysis_md_path, "w", encoding="utf-8") as fmd:
            fmd.write(fallback)
        print(f"[LLMAnalyzer] Fallback research explanation saved to {analysis_md_path}")

