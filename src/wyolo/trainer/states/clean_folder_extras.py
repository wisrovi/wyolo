import os
import shutil
import stat

from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


def _force_unlock(path: str) -> None:
    """Attempt to grant full read/write/execute permissions to path."""
    try:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    except Exception:
        pass


def _onerror_unlock(func, path, exc_info):
    """Callback for shutil.rmtree to unlock permissions and retry on failure."""
    try:
        _force_unlock(path)
        func(path)
    except Exception:
        pass


def _force_remove_file(item_path: str) -> None:
    """Robustly unlink a file or symlink even if write-protected."""
    _force_unlock(item_path)
    try:
        os.unlink(item_path)
    except Exception:
        try:
            os.system(f"rm -f '{item_path}' 2>/dev/null")
        except Exception:
            pass


def _force_purge_directory(folder_path: str) -> None:
    """Recursively purge and recreate a directory tree with permission unlocking."""
    if not os.path.exists(folder_path) and not os.path.islink(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return

    _force_unlock(folder_path)
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for f in files:
            fp = os.path.join(root, f)
            _force_remove_file(fp)
        for d in dirs:
            dp = os.path.join(root, d)
            _force_unlock(dp)
            try:
                shutil.rmtree(dp, onerror=_onerror_unlock)
            except Exception:
                pass
            if os.path.exists(dp):
                try:
                    os.system(f"rm -rf '{dp}' 2>/dev/null")
                except Exception:
                    pass

    try:
        shutil.rmtree(folder_path, onerror=_onerror_unlock)
    except Exception:
        pass

    if os.path.exists(folder_path):
        try:
            os.system(f"rm -rf '{folder_path}' 2>/dev/null")
        except Exception:
            pass

    os.makedirs(folder_path, exist_ok=True)
    _force_unlock(folder_path)


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
            _force_unlock(extras_root)
            for item in os.listdir(extras_root):
                item_path = os.path.join(extras_root, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    _force_remove_file(item_path)

        for folder in self.FOLDER_CLEAN:
            folder_path = os.path.join(project_path, "extras", folder)
            _force_purge_directory(folder_path)

        return ctx
