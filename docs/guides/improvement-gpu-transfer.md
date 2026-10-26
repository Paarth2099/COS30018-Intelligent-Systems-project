# RTX 3070 improvement experiment

Run from your existing project-github folder, with this improvements-training folder inside it.
No existing architecture, GUI, checkpoint or project source file needs replacement.
The package uses its own bundled source; it reads your existing data, original split and configuration files.
Do not copy the laptop .venv. Use the CUDA environment that worked on this PC before.

1. Open the PC project folder in VS Code. Extract improvements-training-package.zip into it.
   Check that improvements-training/train_improvements.py is directly below the project folder,
   not inside another duplicated improvements-training folder.
2. In PowerShell run:

   .\.venv\Scripts\python.exe .\improvements-training\preflight_improvement_training.py

   Continue only if it prints PASS. If the .venv path is different, use your existing CUDA Python.
   A split/config mismatch is a STOP condition, not permission to generate a different split.
3. Run training (8 runs: 2 tasks x 2 seeds x 2 conditions, 5 epochs each):

   .\.venv\Scripts\python.exe .\improvements-training\train_improvements.py --device cuda

   Do not add --smoke. Leave the terminal open and prevent PC sleep. Progress prints task, seed,
   condition and epoch. This is a longer controlled experiment than training one model.
   Completed runs are verified and skipped when restarted. An interrupted partial run is preserved;
   if it stops, send the error before attempting recovery. Do not delete partial evidence.
4. After training exits successfully, run:

   .\.venv\Scripts\python.exe .\improvements-training\evaluate_training_improvements.py --device cuda

   This verifies all eight selections, evaluates official benchmark tests and the SAME 160 prepared
   development images, and writes per-class metrics, confusion matrices and paired predictions.
   It does not load any new final-evaluation photos. A successful evaluation writes verification.json.
5. Send these TWO complete output folders back to this chat/laptop:

   experiments/improvements_gpu_run01
   evidence/improvements/gpu_evaluation_run01

   You can zip them together from the project root:

   Compress-Archive -Path .\experiments\improvements_gpu_run01, .\evidence\improvements\gpu_evaluation_run01 -DestinationPath .\improvements_gpu_results.zip

   Keep the originals. If that zip already exists, choose a new filename rather than overwrite it.

Expected training split sizes: MNIST 54000/6000; EMNIST 628142/69790.
Expected main outputs: 8 selection.json files and 8 checkpoints; evaluation verification with smoke=false,
8 runs, 160 development images and 640 development predictions.
Tiny CPU smoke-test results are plumbing checks only and are not included as trained models here.

If preflight says a project file is missing or differs, the exact three original files are in
required-project-files/ inside this package. Compare/back up the PC copy before copying those files
to their matching project-root paths. These are split/config metadata, not models or architecture files.
Dataset files are not bundled: the PC already used them for the previous successful GPU training.
