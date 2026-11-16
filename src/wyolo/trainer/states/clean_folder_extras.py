import os
import shutil

from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="CleanFolderExtra", version="v1.0")
class CleanFolderExtra:
    """Step to clean up extra output folders before execution."""

    FOLDER_CLEAN = [
        "post_train_results",
        "model_focus",
        "llm",
        "bootstrap",
        "complexity",
        "failures",
        "robustness",
        "paper_table_results",
        "adversarial",
        "cross_domain",
        "feature_space",
        "quantitative_xai",
        "uncertainty",
    ]

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext) -> PostTrainContext:
        """
        Executes the folder cleaning process.

        Args:
            ctx (PostTrainContext): The pipeline context containing the project path.

        Returns:
            PostTrainContext: The updated pipeline context.
        """
        project_path = ctx.project_path

        extras_root = os.path.join(project_path, "extras")
        if os.path.exists(extras_root):
            for item in os.listdir(extras_root):
                item_path = os.path.join(extras_root, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    try:
                        os.remove(item_path)
                    except Exception:
                        pass

        for folder in self.FOLDER_CLEAN:
            folder_path = os.path.join(project_path, "extras", folder)
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path, topdown=False):
                    for f in files:
                        try:
                            os.remove(os.path.join(root, f))
                        except Exception:
                            pass
                    for d in dirs:
                        try:
                            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                        except Exception:
                            pass
                shutil.rmtree(folder_path, ignore_errors=True)

            # Recreate empty folder
            os.makedirs(folder_path, exist_ok=True)

        return ctx
