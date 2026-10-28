# Research commands

Run from the repository root using the project's Python environment:

```powershell
.\.venv\Scripts\python.exe -m research.train_models_week11 --help
.\.venv\Scripts\python.exe -m research.evaluate_models_week11 --help
.\.venv\Scripts\python.exe -m research.train_improvements --help
```

Post-evaluation studies (results are recorded in `results/`):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt   # scikit-learn, research only
.\.venv\Scripts\python.exe -m research.compare_classical_models          # kNN/SVM/forest vs networks; needs data/MNIST
.\.venv\Scripts\python.exe -m research.study_case_rule                   # fit and evaluate the line-case rule
.\.venv\Scripts\python.exe -m research.export_benchmark_confusion        # copy archived confusion matrices for the GUI
```

The [command map](../docs/research-command-map.json) maps former root-level scripts to module commands. Windows wrappers now live in `scripts/`; `run_gui.ps1` stays at the root.

Research commands need the datasets and detailed evidence for their particular study. They are not required to run the GUI. Restore those inputs using the [archive guide](../docs/guides/research-archive.md). Some historical verifiers deliberately require the original source hashes: use the immutable archived source tree for those studies, rather than modifying their controls to accept a different program.

The improvement evaluation requires `--prepared` pointing to the transferred `development_arrays` directory. It must contain the exact prepared arrays and manifest used for the paired comparison. For an original GPU transfer package, pass its directory as `--package` to `research.preflight_improvement_training`; without it, preflight checks the current datasets, splits and device.

Do not run report generators without their archived inputs, or overwrite recorded final results with a development run. Read each command's arguments and source before running it.
