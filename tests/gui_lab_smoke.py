"""Desktop smoke test for the Model lab: real Tk window, real MNIST training.

Run separately (needs an interactive desktop and data/MNIST/raw):
    .venv\\Scripts\\python.exe tests/gui_lab_smoke.py
Trains a small MLP for one epoch, uses it in the main recogniser, checks every
lab tab draws, and saves screenshots plus verification.json under evidence/gui/.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from PIL import ImageGrab

from app import HNRSApp, ROOT
from gui_core import open_photo

out = ROOT / "evidence/gui" / ("lab_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
out.mkdir(parents=True, exist_ok=False)
root = tk.Tk()
app = HNRSApp(root)
root.geometry("1180x860+30+30")
errors, report = [], {"created_utc": datetime.now(timezone.utc).isoformat(), "checks": []}


def pump(condition, seconds=600):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        root.update()
        time.sleep(0.03)
    root.update()
    assert condition(), "Timed out waiting for the GUI"


def screenshot(window, name):
    window.deiconify()
    window.lift()
    window.attributes("-topmost", True)
    window.update()
    time.sleep(0.4)
    window.update()
    x, y = window.winfo_rootx(), window.winfo_rooty()
    ImageGrab.grab(bbox=(x, y, x + window.winfo_width(), y + window.winfo_height())).save(out / name)
    window.attributes("-topmost", False)


def check(text):
    report["checks"].append(text)
    print("PASS:", text, flush=True)


try:
    with patch("app.messagebox.showerror", side_effect=lambda t, m, **k: errors.append(m)), \
            patch("gui_lab.messagebox.showerror", side_effect=lambda t, m, **k: errors.append(m)):
        root.update()
        app.open_lab()
        lab = app.lab
        pump(lambda: lab.benchmark_canvas.find_all())
        lab.window.geometry("1180x820+40+30")
        for index, name in enumerate(["compare", "curves"], 1):
            lab.notebook.select(index)
            lab.refresh_static_views()
            screenshot(lab.window, f"lab_{name}_mnist.png")
        assert len(lab.table.get_children()) == 9, "4 networks + 5 classical baselines expected for MNIST"
        check("Compare tab lists 4 networks and 5 classical baselines on MNIST")
        lab.compare_task.set("EMNIST characters")
        lab.notebook.select(1)
        lab.draw_compare()
        assert len(lab.table.get_children()) == 4
        screenshot(lab.window, "lab_compare_emnist.png")
        lab.curve_task.set("EMNIST characters")
        lab.curve_model.set("resnet18")
        lab.notebook.select(2)
        lab.draw_curves()
        assert "Most frequent test mistakes" in lab.curve_notes.get()
        screenshot(lab.window, "lab_curves_emnist_resnet18.png")
        check("Curves tab draws history and 62-class confusion for EMNIST ResNet-18")

        # Invalid settings are rejected with a message, not an exception.
        lab.notebook.select(0)
        lab.vars["epochs"].set("abc")
        lab.start()
        assert errors.pop() and not lab.busy
        check("Invalid hyperparameters show an error and do not start training")

        settings = dict(task="MNIST digits", model="mlp", epochs="2", learning_rate="0.001",
                        batch_size="64", train_samples="3000", validation_samples="1000", optimizer="adam")
        for key, value in settings.items():
            lab.vars[key].set(value)
        lab.start()
        assert lab.busy and str(lab.start_button["state"]) == "disabled"
        pump(lambda: len(lab.history) >= 1)
        screenshot(lab.window, "lab_training_in_progress.png")
        pump(lambda: not lab.busy)
        assert not errors, errors
        record = lab.record
        assert record and record["checkpoint"] and len(record["history"]) == 2
        assert record["validation_accuracy"] > 0.8, record["validation_accuracy"]
        assert lab.train_confusion.find_all() and str(lab.use_button["state"]) == "normal"
        screenshot(lab.window, "lab_training_complete.png")
        check(f"MLP trained on 3,000 MNIST images for 2 epochs: validation {record['validation_accuracy']:.1%}")

        lab.use_model()
        name = app.model.get()
        assert name.startswith("lab-mlp") and app.mode.get() == "Numbers"
        assert name in app.combos[1]["values"]
        photo = ROOT / "data/segmentation_photos/Manula/Manula_07_label-420.jpg"
        app.set_image(open_photo(photo), {"route": "file", "file": str(photo), "sha256": ""})
        app.recognise()
        pump(lambda: not app.busy)
        assert app.result is not None and app.settings["model"] == name, errors
        assert app.settings["checkpoint_sha256"] == record["checkpoint_sha256"]
        screenshot(root, "main_with_lab_model.png")
        check(f"Lab model selected in main window and recognised {photo.name} as {app.prediction.get()}")

        # Line-case rule in the main window for the report screenshot.
        app.mode.set("Alphanumeric")
        app.mode_changed()
        photo = ROOT / "data/own_strings/development/Manula/Manula_02_text-Room7.jpg"
        app.set_image(open_photo(photo), {"route": "file", "file": str(photo), "sha256": ""})
        app.recognise()
        pump(lambda: not app.busy)
        assert app.prediction.get() == "Room7", app.prediction.get()
        assert app.result["prediction"]["model_text"] == "R00M7"
        app.viewport.yview_moveto(0)
        screenshot(root, "main_line_case_room7.png")
        check("ResNet-18 reads R00M7; line-case rule outputs Room7")
        assert not errors, errors
        report["status"] = "PASS"
        (out / "verification.json").write_text(json.dumps(report, indent=2))
        print("Evidence:", out, flush=True)
except Exception as error:
    report.update(status="FAIL", error=repr(error), dialogs=errors)
    (out / "verification.json").write_text(json.dumps(report, indent=2))
    raise
finally:
    if app.lab is not None:
        app.lab.busy = False
        app.lab.close()
    if not app.closed:
        app.close()
