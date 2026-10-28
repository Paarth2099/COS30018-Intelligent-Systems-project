"""Model lab window: train models with chosen hyperparameters and visualise results.

Three tabs:
  Train a model     - dataset, architecture and hyperparameters; live accuracy
                      and loss curves; validation confusion matrix; the trained
                      model can then be selected in the main recogniser.
  Compare models    - recorded official-test benchmarks (networks and classical
                      baselines) and the 120-photo final evaluation.
  Curves and errors - training history and official-test confusion matrix of
                      each shipped checkpoint, with its most frequent mistakes.

Training runs in a worker thread; a queue passes progress to the Tk thread.
"""
import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import gui_charts
from gui_training import ARCHITECTURES, OPTIMIZERS, TASKS, TrainingSettings, dataset_available, load_task_data, train

MODELS = ("mlp", "cnn", "resnet18", "mobilenet_v3_small")
MODE_FOR_TASK = {"mnist": "Numbers", "emnist": "Alphanumeric"}
DEFAULT_FOR_TASK = {"mnist": "cnn", "emnist": "resnet18"}
TASK_LABELS = {"MNIST digits": "mnist", "EMNIST characters": "emnist"}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def shipped_history(root, task, model):
    """Per-epoch rows of a shipped checkpoint's recorded training history."""
    if model in ("mlp", "cnn"):
        folder = "experiments/baseline" if task == "mnist" else "experiments/emnist_byclass/run_01"
        rows = [r for r in read_csv(Path(root) / folder / "history.csv") if r.get("model") == model]
    else:
        rows = read_csv(Path(root) / "experiments/week11_gpu_run01" / task / model / "history.csv")
    return [{key: float(value) for key, value in row.items() if key != "model"} for row in rows]


def top_confusions(matrix, classes, count=8):
    pairs = [(matrix[r][c], classes[r], classes[c]) for r in range(len(matrix))
             for c in range(len(matrix)) if r != c and matrix[r][c]]
    return sorted(pairs, reverse=True)[:count]


class ModelLab:
    def __init__(self, app):
        self.app, self.root = app, app.root
        self.window = tk.Toplevel(self.root)
        self.window.title("HNRS | Model lab")
        width = min(1180, self.window.winfo_screenwidth() - 60)
        height = min(820, self.window.winfo_screenheight() - 100)
        self.window.geometry(f"{width}x{height}+40+30")
        self.window.minsize(860, 600)
        self.window.configure(bg="#f3f6fb")
        self.messages = queue.Queue()
        self.stop_event = threading.Event()
        self.busy = False
        self.data_cache = {}
        self.history = []
        self.batch_losses = []
        self.record = None
        self.closed = False
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)
        self.notebook = notebook
        self.build_train_tab(notebook)
        self.build_compare_tab(notebook)
        self.build_curves_tab(notebook)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = self.window.after(100, self.poll)
        self.window.after(200, self.refresh_static_views)

    # ------------------------------------------------------------------ training
    def build_train_tab(self, notebook):
        tab = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        notebook.add(tab, text="Train a model")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)
        form = ttk.Frame(tab, style="Panel.TFrame", width=270)
        form.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        self.vars = {
            "task": tk.StringVar(value="MNIST digits"), "model": tk.StringVar(value="cnn"),
            "epochs": tk.StringVar(value="3"), "learning_rate": tk.StringVar(value="0.001"),
            "batch_size": tk.StringVar(value="128"), "train_samples": tk.StringVar(value="10000"),
            "validation_samples": tk.StringVar(value="3000"), "optimizer": tk.StringVar(value="adam"),
            "seed": tk.StringVar(value="42")}
        self.download = tk.BooleanVar(value=False)
        ttk.Label(form, text="HYPERPARAMETERS", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.inputs = []
        fields = [("Dataset", "task", list(TASK_LABELS), "readonly"),
                  ("Architecture", "model", list(ARCHITECTURES), "readonly"),
                  ("Optimizer", "optimizer", list(OPTIMIZERS), "readonly"),
                  ("Learning rate", "learning_rate", ["0.1", "0.01", "0.001", "0.0005", "0.0001"], "normal"),
                  ("Batch size", "batch_size", ["32", "64", "128", "256", "512"], "normal")]
        for label, key, values, state in fields:
            ttk.Label(form, text=label, style="Caption.TLabel").pack(anchor="w")
            combo = ttk.Combobox(form, textvariable=self.vars[key], values=values, state=state, width=24)
            combo.pack(fill="x", pady=(1, 4))
            self.inputs.append((combo, state))
        for label, key, low, high, step in [("Epochs", "epochs", 1, 50, 1),
                                            ("Training samples", "train_samples", 100, 698000, 1000),
                                            ("Validation samples", "validation_samples", 100, 69790, 500),
                                            ("Random seed", "seed", 0, 9999, 1)]:
            ttk.Label(form, text=label, style="Caption.TLabel").pack(anchor="w")
            spin = ttk.Spinbox(form, textvariable=self.vars[key], from_=low, to=high, increment=step, width=24)
            spin.pack(fill="x", pady=(1, 4))
            self.inputs.append((spin, "normal"))
        self.vars["task"].trace_add("write", lambda *_: self.update_availability())
        check = ttk.Checkbutton(form, text="Download dataset if missing", variable=self.download)
        check.pack(anchor="w", pady=(4, 2))
        self.inputs.append((check, "normal"))
        self.availability = tk.StringVar()
        ttk.Label(form, textvariable=self.availability, style="Caption.TLabel", wraplength=250).pack(anchor="w")
        buttons = ttk.Frame(form, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(12, 0))
        self.start_button = ttk.Button(buttons, text="Start training", style="Accent.TButton", command=self.start)
        self.start_button.pack(fill="x")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(fill="x", pady=(6, 0))
        self.use_button = ttk.Button(buttons, text="Use in recogniser", command=self.use_model, state="disabled")
        self.use_button.pack(fill="x", pady=(6, 0))

        right = ttk.Frame(tab, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        self.train_status = tk.StringVar(value="Choose settings and start training.")
        ttk.Label(right, textvariable=self.train_status, style="Panel.TLabel", wraplength=780).grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.progress = ttk.Progressbar(right, mode="determinate", maximum=1.0)
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 10))
        self.accuracy_canvas = tk.Canvas(right, bg="white", height=230, width=360, highlightthickness=0)
        self.accuracy_canvas.grid(row=2, column=0, sticky="nsew", padx=(0, 6))
        self.loss_canvas = tk.Canvas(right, bg="white", height=230, width=360, highlightthickness=0)
        self.loss_canvas.grid(row=2, column=1, sticky="nsew", padx=(6, 0))
        self.train_confusion = tk.Canvas(right, bg="white", height=300, width=360, highlightthickness=0)
        self.train_confusion.grid(row=3, column=0, sticky="nsew", pady=(10, 0), padx=(0, 6))
        self.train_summary = tk.Text(right, height=14, width=40, relief="flat", bg="#f8fafc", fg="#192b46",
                                     font=("Segoe UI", 9), wrap="word")
        self.train_summary.grid(row=3, column=1, sticky="nsew", pady=(10, 0), padx=(6, 0))
        self.train_summary.insert("1.0", "Training uses the project's fixed train/validation split. The official "
                                         "test sets are never loaded here, so lab accuracy is a validation figure.\n\n"
                                         "Tip: an MLP or CNN on 10,000 MNIST images trains in seconds to a minute "
                                         "on a CPU; ResNet-18 and EMNIST take much longer.")
        self.update_availability()

    def update_availability(self):
        task = TASK_LABELS[self.vars["task"].get()]
        if dataset_available(self.app.service.root, task):
            self.availability.set(f"{TASKS[task]}: found in data/.")
        else:
            size = "about 60 MB" if task == "mnist" else "about 560 MB"
            self.availability.set(f"{TASKS[task]}: not found in data/. Tick download ({size}) to fetch it.")

    def read_settings(self):
        try:
            values = {key: var.get().strip() for key, var in self.vars.items()}
            return TrainingSettings(
                task=TASK_LABELS[values["task"]], model=values["model"], epochs=int(values["epochs"]),
                learning_rate=float(values["learning_rate"]), batch_size=int(values["batch_size"]),
                train_samples=int(values["train_samples"]), validation_samples=int(values["validation_samples"]),
                optimizer=values["optimizer"], seed=int(values["seed"])).validate()
        except ValueError as error:
            text = str(error)
            raise ValueError(text if text.endswith(".") else "Enter whole numbers for epochs, samples, "
                             "batch size and seed, and a decimal learning rate.") from error

    def set_busy(self, busy):
        self.busy = busy
        for widget, state in self.inputs:
            widget.configure(state="disabled" if busy else state)
        self.start_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        if busy:
            self.use_button.configure(state="disabled")

    def start(self):
        if self.busy:
            return
        try:
            settings = self.read_settings()
        except ValueError as error:
            messagebox.showerror("Check the settings", str(error), parent=self.window)
            return
        root = self.app.service.root
        if not dataset_available(root, settings.task) and not self.download.get():
            messagebox.showerror("Dataset missing", self.availability.get(), parent=self.window)
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = root / "experiments/gui_runs" / f"{stamp}_{settings.task}_{settings.model}"
        self.history, self.batch_losses, self.record = [], [], None
        self.stop_event.clear()
        self.set_busy(True)
        self.progress.configure(value=0)
        self.train_summary.delete("1.0", "end")
        self.draw_training()
        gui_charts.heatmap(self.train_confusion, [], [], "Validation confusion matrix")
        download = self.download.get()

        def worker():
            try:
                if settings.task not in self.data_cache:
                    self.messages.put(("status", f"Loading {TASKS[settings.task]}..."))
                    self.data_cache[settings.task] = load_task_data(root, settings.task, download)
                self.messages.put(("status", f"Training {settings.model} on {settings.train_samples:,} samples..."))
                record = train(settings, self.data_cache[settings.task], output,
                               progress=lambda event: self.messages.put(("progress", event)),
                               should_stop=self.stop_event.is_set)
                self.messages.put(("done", record))
            except Exception as error:  # Reported in the window; the GUI must stay usable.
                self.messages.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        self.train_status.set("Stopping after the current batch...")

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, value = self.messages.get_nowait()
                self.handle(kind, value)
        except queue.Empty:
            pass
        self.poll_id = self.window.after(100, self.poll)

    def handle(self, kind, value):
        if kind == "status":
            self.train_status.set(value)
        elif kind == "error":
            self.set_busy(False)
            self.train_status.set("Training failed: " + value)
            messagebox.showerror("Training failed", value, parent=self.window)
        elif kind == "progress" and value["kind"] == "batch":
            epochs = int(self.vars["epochs"].get())
            done = (value["epoch"] - 1 + value["batch"] / value["batches"]) / epochs
            self.progress.configure(value=done)
            self.batch_losses.append((value["epoch"] - 1 + value["batch"] / value["batches"], value["loss"]))
            self.train_status.set(f"Epoch {value['epoch']}/{epochs} | batch {value['batch']}/{value['batches']} "
                                  f"| loss {value['loss']:.4f}")
            if value["batch"] % 50 == 0 or value["batch"] == value["batches"]:
                self.draw_training()
        elif kind == "progress" and value["kind"] == "epoch":
            self.history.append(value)
            self.draw_training()
        elif kind == "done":
            self.finish(value)

    def draw_training(self):
        epochs = [(row["epoch"], row) for row in self.history]
        gui_charts.line_chart(self.accuracy_canvas, {
            "train": [(e, r["train_accuracy"]) for e, r in epochs],
            "validation": [(e, r["validation_accuracy"]) for e, r in epochs]},
            "Accuracy per epoch", y_label="accuracy")
        # Same series order as the accuracy chart, so train/validation colours match.
        gui_charts.line_chart(self.loss_canvas, {
            "train": [(e, r["train_loss"]) for e, r in epochs],
            "validation": [(e, r["validation_loss"]) for e, r in epochs],
            "batch loss": self.batch_losses},
            "Cross-entropy loss", y_label="loss")

    def finish(self, record):
        self.set_busy(False)
        self.record = record
        self.draw_training()
        if not record.get("checkpoint"):
            self.train_status.set("Stopped before the first epoch finished; no model was saved.")
            return
        self.progress.configure(value=1.0)
        classes = record["classes"]
        gui_charts.heatmap(self.train_confusion, record["confusion_matrix"], classes,
                           "Validation confusion matrix (best epoch)")
        config = record["config"]
        lines = [f"{'Stopped early' if record['stopped'] else 'Finished'} in {record['training_seconds']:.1f} s "
                 f"on {config['device']}.",
                 f"Best epoch {record['best_epoch']}: validation accuracy {record['validation_accuracy']:.2%} "
                 f"on {config['validation_count']:,} images.",
                 f"Settings: {config['model']}, {config['optimizer']}, lr {config['learning_rate']}, "
                 f"batch {config['batch_size']}, {config['train_count']:,} training samples, seed {config['seed']}.",
                 "", "Most frequent validation mistakes (true -> predicted):"]
        lines += [f"  {t} -> {p}: {n}" for n, t, p in top_confusions(record["confusion_matrix"], classes)]
        lines += ["", f"Saved to {Path(record['checkpoint']).parent}"]
        self.train_summary.insert("1.0", "\n".join(lines))
        self.train_status.set("Training complete. Select 'Use in recogniser' to try it on photos.")
        self.use_button.configure(state="normal")

    def use_model(self):
        if not self.record or not self.record.get("checkpoint"):
            return
        config = self.record["config"]
        mode = MODE_FOR_TASK[config["task"]]
        name = f"lab-{config['model']}-{Path(self.record['checkpoint']).parent.name[:15]}"
        try:
            self.app.service.register_custom(mode, name, self.record["checkpoint"], self.record["checkpoint_sha256"])
        except ValueError as error:
            messagebox.showerror("Cannot use model", str(error), parent=self.window)
            return
        self.app.use_custom_model(mode, name)
        self.train_status.set(f"'{name}' is now selectable in {mode} mode of the main window.")

    # --------------------------------------------------------------- comparison
    def build_compare_tab(self, notebook):
        tab = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        notebook.add(tab, text="Compare models")
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)
        top = ttk.Frame(tab, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.compare_task = tk.StringVar(value="MNIST digits")
        for label in TASK_LABELS:
            ttk.Radiobutton(top, text=label, value=label, variable=self.compare_task,
                            command=self.draw_compare).pack(side="left", padx=(0, 12))
        ttk.Label(top, text="Recorded results; nothing is re-run. Highlighted: application default.",
                  style="Caption.TLabel").pack(side="left", padx=12)
        self.benchmark_canvas = tk.Canvas(tab, bg="white", height=280, width=420, highlightthickness=0)
        self.benchmark_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=8)
        self.photo_canvas = tk.Canvas(tab, bg="white", height=280, width=420, highlightthickness=0)
        self.photo_canvas.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=8)
        columns = ("model", "kind", "accuracy", "macro_f1", "size", "speed")
        self.table = ttk.Treeview(tab, columns=columns, show="headings", height=9)
        for column, heading, width in zip(columns, ["Model", "Type", "Test accuracy", "Macro F1",
                                                    "Parameters / size", "ms per image"],
                                          [170, 260, 110, 90, 140, 100]):
            self.table.heading(column, text=heading)
            self.table.column(column, width=width, anchor="w" if column in ("model", "kind") else "e")
        self.table.grid(row=2, column=0, columnspan=2, sticky="nsew")

    def benchmark_rows(self, task):
        root = self.app.service.root
        rows = []
        for row in read_csv(root / "results/four_models/benchmark_summary.csv"):
            if row["task"] == task:
                rows.append(dict(model=row["model"], kind="Neural network", accuracy=float(row["accuracy"]),
                                 macro_f1=float(row["macro_f1"]), size=f"{int(row['parameters']):,} params",
                                 speed=float(row["ms_per_image"])))
        if task == "mnist":
            for row in read_csv(root / "results/classical_baselines/benchmark.csv"):
                rows.append(dict(model=row["family"], kind=row["description"], accuracy=float(row["test_accuracy"]),
                                 macro_f1=float(row["macro_f1"]),
                                 size=f"{int(row['model_bytes']) / 1e6:.1f} MB model",
                                 speed=float(row["ms_per_digit"])))
        return rows

    def draw_compare(self):
        task = TASK_LABELS[self.compare_task.get()]
        rows = sorted(self.benchmark_rows(task), key=lambda r: r["accuracy"], reverse=True)
        lowest = min((r["accuracy"] for r in rows), default=0)
        gui_charts.bar_chart(self.benchmark_canvas, [r["model"] for r in rows], [r["accuracy"] for r in rows],
                             f"Official {task.upper()} test accuracy", highlight={DEFAULT_FOR_TASK[task]},
                             y_range=(max(0, lowest - 0.05), 1.0))
        self.table.delete(*self.table.get_children())
        for r in rows:
            self.table.insert("", "end", values=(r["model"], r["kind"], f"{r['accuracy']:.2%}",
                                                 f"{r['macro_f1']:.4f}", r["size"], f"{r['speed']:.3f}"))
        collections = ("single_digits", "numbers") if task == "mnist" else ("single_letters", "strings")
        path = self.app.service.root / "results/final_evaluation/summary.json"
        summary = json.loads(path.read_text()) if path.exists() else []
        photo = [r for r in summary if r["collection"] in collections
                 and r["method"] in ("components", "grouped_components")]
        labels = [f"{r['model']} | {r['collection'].replace('_', ' ')}" for r in photo]
        gui_charts.bar_chart(self.photo_canvas, labels, [r["accuracy"] for r in photo],
                             "Final 120-photo evaluation: exact accuracy",
                             highlight={f"{DEFAULT_FOR_TASK[task]} | {c.replace('_', ' ')}" for c in collections},
                             y_range=(0, 1.0))

    # ------------------------------------------------------- curves and errors
    def build_curves_tab(self, notebook):
        tab = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        notebook.add(tab, text="Curves and errors")
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)
        top = ttk.Frame(tab, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.curve_task = tk.StringVar(value="MNIST digits")
        self.curve_model = tk.StringVar(value="cnn")
        for label, variable, values in [("Dataset", self.curve_task, list(TASK_LABELS)),
                                        ("Shipped model", self.curve_model, list(MODELS))]:
            ttk.Label(top, text=label, style="Caption.TLabel").pack(side="left", padx=(0, 4))
            combo = ttk.Combobox(top, textvariable=variable, values=values, state="readonly", width=20)
            combo.pack(side="left", padx=(0, 14))
            combo.bind("<<ComboboxSelected>>", lambda e: self.draw_curves())
        self.curve_accuracy = tk.Canvas(tab, bg="white", height=230, width=420, highlightthickness=0)
        self.curve_accuracy.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=8)
        self.curve_loss = tk.Canvas(tab, bg="white", height=230, width=420, highlightthickness=0)
        self.curve_loss.grid(row=2, column=0, sticky="nsew", padx=(0, 6))
        self.curve_confusion = tk.Canvas(tab, bg="white", height=420, width=420, highlightthickness=0)
        self.curve_confusion.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=(6, 0), pady=(8, 0))
        self.curve_notes = tk.StringVar()
        ttk.Label(tab, textvariable=self.curve_notes, style="Caption.TLabel", wraplength=1000).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.confusion_record = None

    def draw_curves(self):
        task, model = TASK_LABELS[self.curve_task.get()], self.curve_model.get()
        rows = shipped_history(self.app.service.root, task, model)
        gui_charts.line_chart(self.curve_accuracy, {
            "train": [(r["epoch"], r["train_accuracy"]) for r in rows],
            "validation": [(r["epoch"], r["validation_accuracy"]) for r in rows]},
            f"{model} on {task.upper()}: accuracy", y_label="accuracy")
        gui_charts.line_chart(self.curve_loss, {
            "train": [(r["epoch"], r["train_loss"]) for r in rows],
            "validation": [(r["epoch"], r["validation_loss"]) for r in rows]},
            "Cross-entropy loss", y_label="loss")
        if self.confusion_record is None:
            path = self.app.service.root / "results/four_models/benchmark_confusion.json"
            self.confusion_record = json.loads(path.read_text())["matrices"] if path.exists() else {}
        entry = self.confusion_record.get(f"{task}/{model}")
        if not entry:
            gui_charts.heatmap(self.curve_confusion, [], [], "Official test confusion matrix")
            self.curve_notes.set("Confusion data not found: run python -m research.export_benchmark_confusion.")
            return
        if task == "mnist":
            classes = [str(i) for i in range(10)]
        else:
            from hnrs_chars.data import CLASSES as classes
        matrix = entry["confusion_matrix"]
        gui_charts.heatmap(self.curve_confusion, matrix, classes,
                           f"Official test confusion ({entry['correct']:,}/{entry['count']:,} correct)")
        mistakes = ", ".join(f"{t}->{p} ({n:,})" for n, t, p in top_confusions(matrix, classes, 6))
        self.curve_notes.set(f"Most frequent test mistakes (true->predicted): {mistakes}. "
                             "Shading is row-normalised: green diagonal = recall, red = confusion rate.")

    def refresh_static_views(self):
        if not self.closed:
            self.draw_training()
            self.draw_compare()
            self.draw_curves()

    def close(self):
        if self.busy and not messagebox.askyesno(
                "Training in progress", "Stop training and close the Model lab?", parent=self.window):
            return
        self.stop_event.set()
        self.closed = True
        self.window.after_cancel(self.poll_id)
        self.window.destroy()
        self.app.lab = None
