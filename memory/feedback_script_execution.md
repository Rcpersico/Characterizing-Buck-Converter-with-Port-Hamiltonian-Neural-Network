---
name: feedback-script-execution
description: Always use the conda run command from notes.md; never invoke python directly or install packages
metadata:
  type: feedback
---

Always use this exact command form to run Python scripts:

    & "$env:USERPROFILE\anaconda3\Scripts\conda.exe" run -n nilm-gpu python <script_path>

**Why:** The project uses a specific conda environment (`nilm-gpu`) that has all required packages (PyTorch, CUDA, matplotlib, etc.). Direct `python` invocation uses the wrong environment.

**How to apply:** Any time a script needs to be executed, use this form. If a package is missing, ask the user before attempting any install — never run pip or conda install unilaterally.
