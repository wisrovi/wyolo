import os
import glob
import re

STATES_DIR = "/home/william.rodriguez/Documents/w_libraries/train_service2/wyoloservice2_worker/executor_v2.0/wtrain/lib/src/wyolo/trainer/states"

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # The previous script injected:
    # import sys
        md_content = inspect.cleandoc(sys.modules[__name__].__doc__ or "No description available.")
        md_content = f"""# Analysis Report

{md_content}

## Methodology
This directory contains the outputs and results of this specific analysis. The metrics and plots generated here reflect the model's behavior according to the described methodology."""
    # I want to change it to use sys.modules[self.__module__].__doc__ or the global __doc__
    
    # Let's replace self.__class__.__doc__ with sys.modules[self.__module__].__doc__
    # Wait, the simplest way inside a method is `sys.modules[__name__].__doc__` if it's in the same file.
    
    pattern = re.compile(r'md_content = inspect\.cleandoc\(self\.__class__\.__doc__ or "No description available\."\)')
    replacement = """import sys\n        md_content = inspect.cleandoc(sys.modules[__name__].__doc__ or "No description available.")\n        md_content = f"""# Analysis Report\\n\\n{md_content}\\n\\n## Methodology\\nThis directory contains the outputs and results of this specific analysis. The metrics and plots generated here reflect the model's behavior according to the described methodology.\""""
    
    if pattern.search(content):
        content = pattern.sub(replacement, content)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {filepath}")

for f in glob.glob(os.path.join(STATES_DIR, "*.py")):
    if f.endswith("__init__.py") or "llm_analyzer" in f or "clean_folder_extras" in f or "post_train" in f:
        continue
    process_file(f)
