"""Saniru's local HNRS desktop GUI. Run with the project Python environment."""

import hashlib
import base64
from io import BytesIO
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image
from gui_core import (
    open_photo,
    generate_number,
    generate_text,
    annotated_image,
    export_result,
    RecognitionService,
    MODEL_NAMES,
    DEFAULT_MODELS,
)

ROOT = Path(__file__).resolve().parent


def fit_window(root):
    """Initial client size in Tk's DPI-virtualised coordinates, with frame margin."""
    left, top, right, bottom = 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()
    try:
        import ctypes
        from ctypes import wintypes

        area = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(area), 0):
            left, top, right, bottom = area.left, area.top, area.right, area.bottom
    except (AttributeError, OSError):
        pass  # Non-Windows Tk uses the screen rectangle.
    width = min(1240, max(320, right - left - 32))
    height = min(860, max(240, bottom - top - 80))
    root.geometry(f"{width}x{height}+{left+16}+{top+16}")
    root.minsize(min(640, width), min(400, height))
    return (left, top, right, bottom)


def tk_image(image, master):
    # Use Tk's PNG decoder: Pillow's optional _imagingtk DLL may fail in long
    # Windows checkout paths even when Pillow's regular image functions work.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return tk.PhotoImage(
        master=master, data=base64.b64encode(buffer.getvalue()), format="png"
    )


# Main desktop GUI for image input, recognition controls and result display.
class HNRSApp:
    def __init__(self, root):
        self.root = root
        self.image = None
        self.origin = None
        self.result = None
        self.settings = None
        self.service = RecognitionService(ROOT)
        self.busy = False
        self.closed = False
        self.lab = None
        self.messages = queue.Queue()
        self.photos = []
        self.job_thread = None
        root.title("HNRS | Handwriting Studio")
        self.work_area = fit_window(root)
        root.configure(bg="#f3f6fb")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background="#f3f6fb")
        style.configure("TLabel", background="#f3f6fb", foreground="#192b46")
        style.configure("Panel.TFrame", background="white")
        style.configure("Panel.TLabel", background="white", foreground="#192b46")
        style.configure(
            "Caption.TLabel",
            background="white",
            foreground="#64748b",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Section.TLabel",
            background="white",
            foreground="#334155",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))
        style.configure("Muted.TLabel", foreground="#64748b")
        style.configure(
            "TButton",
            padding=(12, 9),
            background="#eef3fa",
            foreground="#26415e",
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TButton",
            background=[("active", "#dfe8f4")],
            foreground=[("disabled", "#94a3b8")],
        )
        style.configure(
            "Accent.TButton", background="#2563eb", foreground="white", padding=(12, 13)
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", "#dbe5f5"), ("active", "#1d4ed8")],
            foreground=[("disabled", "#8192ab"), ("!disabled", "white")],
        )
        style.configure(
            "TEntry",
            padding=7,
            fieldbackground="#f8fafc",
            bordercolor="#dce4ef",
            lightcolor="#dce4ef",
            darkcolor="#dce4ef",
        )
        style.configure(
            "TCombobox",
            padding=6,
            fieldbackground="#f8fafc",
            background="#eef3fa",
            bordercolor="#dce4ef",
            arrowcolor="#475569",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#f8fafc")],
            selectbackground=[("readonly", "#f8fafc")],
            selectforeground=[("readonly", "#192b46")],
        )
        style.configure("TNotebook", background="white", borderwidth=0)
        style.configure(
            "TNotebook.Tab", padding=(12, 8), background="#f1f5f9", foreground="#64748b"
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#eaf1ff")],
            foreground=[("selected", "#1d4ed8")],
        )
        style.configure(
            "Horizontal.TScrollbar",
            background="#dbe3ee",
            troughcolor="white",
            borderwidth=0,
            arrowsize=12,
        )
        style.configure("TCheckbutton", background="white", foreground="#192b46")
        style.configure("TRadiobutton", background="white", foreground="#192b46")
        style.map("TCheckbutton", background=[("active", "white")])
        style.map("TRadiobutton", background=[("active", "white")])
        style.configure(
            "Horizontal.TProgressbar",
            background="#2563eb",
            troughcolor="#e2e8f0",
            borderwidth=0,
        )
        self.mode = tk.StringVar(value="Numbers")
        self.model = tk.StringVar(value="cnn")
        self.method = tk.StringVar(value="components")
        self.representation = tk.StringVar(value="otsu")
        self.source_hint = tk.StringVar(
            value="Use one line of dark digits on pale paper."
        )
        self.folder_hint = tk.StringVar(value="Single-digit photo folder")
        self.request_hint = tk.StringVar(value="Number to create  ·  1–12 digits")
        self.result_heading = tk.StringVar(value="RECOGNISED NUMBER")
        self.regions_heading = tk.StringVar(value="Detected digits")
        self.crops_heading = tk.StringVar(value="Digit breakdown")
        self.folder = tk.StringVar(value=str(ROOT / "data/segmentation_photos"))
        self.number = tk.StringVar(value="007")
        self.status = tk.StringVar(value="Load a photo or generate a number to begin.")
        self.source_text = tk.StringVar(value="No image selected")
        self.prediction = tk.StringVar(value="No prediction yet")
        self.details = tk.StringVar(value="Your recognition result will appear here.")

        # Keep the layout's readable content size while allowing smaller windows.
        # Scrollbars and keyboard focus make every control reachable at high DPI.
        shell = ttk.Frame(root)
        shell.pack(fill="both", expand=True)
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)
        self.viewport = tk.Canvas(
            shell,
            bg="#f3f6fb",
            highlightthickness=0,
            xscrollincrement=24,
            yscrollincrement=24,
        )
        self.viewport.grid(row=0, column=0, sticky="nsew")
        self.page_vscroll = ttk.Scrollbar(
            shell, orient="vertical", command=self.viewport.yview
        )
        self.page_vscroll.grid(row=0, column=1, sticky="ns")
        self.page_hscroll = ttk.Scrollbar(
            shell, orient="horizontal", command=self.viewport.xview
        )
        self.page_hscroll.grid(row=1, column=0, sticky="ew")
        self.viewport.configure(
            yscrollcommand=self.page_vscroll.set, xscrollcommand=self.page_hscroll.set
        )
        frame = ttk.Frame(self.viewport, padding=24)
        self.page = frame
        self.page_item = self.viewport.create_window((0, 0), window=frame, anchor="nw")
        self.viewport.bind("<Configure>", self.resize_page)
        header = ttk.Frame(frame)
        header.pack(fill="x", pady=(0, 22))
        brand = tk.Label(
            header,
            text=" HN ",
            bg="#2563eb",
            fg="white",
            font=("Segoe UI", 20, "bold"),
            padx=8,
            pady=8,
        )
        brand.pack(side="left", padx=(0, 14))
        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text="Handwriting Studio", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            titles,
            text="Recognise handwritten numbers, letters and mixed strings.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        tk.Label(
            header,
            text="COS30018  /  HNRS",
            bg="#e7eef9",
            fg="#43618a",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=8,
        ).pack(side="right")

        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body, style="Panel.TFrame", padding=18, width=280)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 20))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        ttk.Label(sidebar, text="01  /  IMAGE SOURCE", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.routes = ttk.Notebook(sidebar)
        self.routes.grid(row=1, column=0, sticky="ew")
        load_tab = ttk.Frame(self.routes, style="Panel.TFrame", padding=(0, 8))
        gen_tab = ttk.Frame(self.routes, style="Panel.TFrame", padding=(0, 8))
        self.routes.add(load_tab, text="Open photo")
        self.routes.add(gen_tab, text="Generate")
        ttk.Label(
            load_tab, text="Choose a handwriting photo", style="Section.TLabel"
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            load_tab,
            textvariable=self.source_hint,
            style="Caption.TLabel",
            wraplength=232,
        ).pack(anchor="w", pady=(0, 16))
        self.load_button = ttk.Button(
            load_tab, text="Choose image...", command=self.load_dialog
        )
        self.load_button.pack(fill="x")
        ttk.Label(load_tab, text="PNG, JPG, BMP or TIFF", style="Caption.TLabel").pack(
            anchor="w", pady=(10, 0)
        )
        ttk.Label(gen_tab, textvariable=self.folder_hint, style="Caption.TLabel").pack(
            anchor="w"
        )
        folder_row = ttk.Frame(gen_tab, style="Panel.TFrame")
        folder_row.pack(fill="x", pady=(5, 9))
        self.folder_entry = ttk.Entry(folder_row, textvariable=self.folder, width=12)
        self.folder_entry.pack(side="left", fill="x", expand=True)
        self.folder_button = ttk.Button(
            folder_row, text="Browse", command=self.folder_dialog
        )
        self.folder_button.pack(side="right", padx=(5, 0))
        ttk.Label(gen_tab, textvariable=self.request_hint, style="Caption.TLabel").pack(
            anchor="w"
        )
        self.number_entry = ttk.Entry(gen_tab, textvariable=self.number)
        self.number_entry.pack(fill="x", pady=(5, 9))
        self.generate_button = ttk.Button(
            gen_tab, text="Generate image", command=self.generate
        )
        self.generate_button.pack(fill="x")
        ttk.Separator(sidebar).grid(row=2, column=0, sticky="ew", pady=(4, 10))
        ttk.Label(sidebar, text="02  /  RECOGNITION", style="Section.TLabel").grid(
            row=3, column=0, sticky="w", pady=(0, 6)
        )
        options = ttk.Frame(sidebar, style="Panel.TFrame")
        options.grid(row=4, column=0, sticky="ew")
        self.combos = []
        for label, variable, values in [
            ("Recognition mode", self.mode, ["Numbers", "Alphanumeric"]),
            ("Recognition model", self.model, MODEL_NAMES),
            ("Character separation", self.method, ["components", "projection"]),
            ("Image preparation", self.representation, ["otsu", "fixed", "grayscale"]),
        ]:
            ttk.Label(options, text=label, style="Caption.TLabel").pack(
                anchor="w", pady=(0, 4)
            )
            combo = ttk.Combobox(
                options,
                textvariable=variable,
                values=values,
                state="readonly",
                width=20,
            )
            combo.pack(fill="x", pady=(0, 4))
            combo.bind(
                "<<ComboboxSelected>>",
                self.mode_changed if variable is self.mode else self.settings_changed,
            )
            self.combos.append(combo)
        # Line-context letter case (hnrs_chars/case_rule.py): photos in Alphanumeric mode only.
        self.line_case = tk.BooleanVar(value=True)
        self.case_check = ttk.Checkbutton(
            options,
            text="Letter case from line context",
            variable=self.line_case,
            command=self.settings_changed,
        )
        self.case_check.pack(anchor="w", pady=(2, 2))
        self.recognise_button = ttk.Button(
            sidebar,
            text="Recognise number",
            style="Accent.TButton",
            command=self.recognise,
        )
        self.recognise_button.grid(row=5, column=0, sticky="ew", pady=(4, 4))
        self.export_button = ttk.Button(
            sidebar, text="Export result...", command=self.export_dialog
        )
        self.export_button.grid(row=6, column=0, sticky="ew")
        self.lab_button = ttk.Button(
            sidebar, text="Model lab: train & compare...", command=self.open_lab
        )
        self.lab_button.grid(row=7, column=0, sticky="ew", pady=(4, 0))

        workspace = ttk.Frame(body)
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(1, weight=1)
        ttk.Label(
            workspace,
            textvariable=self.source_text,
            style="Muted.TLabel",
            wraplength=620,
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        panes = ttk.Frame(workspace)
        panes.grid(row=1, column=0, sticky="nsew")
        panes.columnconfigure(0, weight=1, uniform="panes")
        panes.columnconfigure(1, weight=1, uniform="panes")
        panes.rowconfigure(0, weight=1)
        for column, title, subtitle, attribute in [
            (0, "Original image", "Your handwriting input", "input_canvas"),
            (1, "Detected digits", "Separated from left to right", "output_canvas"),
        ]:
            card = ttk.Frame(panes, style="Panel.TFrame", padding=14)
            card.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0, 7) if column == 0 else (7, 0),
            )
            ttk.Label(
                card,
                **(
                    {"textvariable": self.regions_heading}
                    if column == 1
                    else {"text": title}
                ),
                style="Section.TLabel",
            ).pack(anchor="w")
            ttk.Label(card, text=subtitle, style="Caption.TLabel").pack(
                anchor="w", pady=(3, 12)
            )
            canvas = tk.Canvas(
                card, bg="#f8fafc", height=100, width=100, highlightthickness=0
            )
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Configure>", lambda e: self.redraw())
            setattr(self, attribute, canvas)
        result_frame = tk.Frame(workspace, bg="#eaf1ff", padx=20, pady=14)
        result_frame.grid(row=2, column=0, sticky="ew", pady=(16, 16))
        tk.Label(
            result_frame,
            textvariable=self.result_heading,
            bg="#eaf1ff",
            fg="#43618a",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        self.prediction_label = tk.Label(
            result_frame,
            textvariable=self.prediction,
            bg="#eaf1ff",
            fg="#1747a6",
            font=("Segoe UI", 29, "bold"),
            wraplength=580,
            justify="left",
        )
        self.prediction_label.pack(anchor="w", pady=(3, 2))
        result_frame.bind(
            "<Configure>",
            lambda e: self.prediction_label.configure(
                wraplength=max(100, e.width - 40)
            ),
        )
        tk.Label(
            result_frame,
            textvariable=self.details,
            bg="#eaf1ff",
            fg="#526b91",
            font=("Segoe UI", 9),
            wraplength=600,
            justify="left",
        ).pack(anchor="w")
        digit_card = ttk.Frame(workspace, style="Panel.TFrame", padding=(14, 12))
        digit_card.grid(row=3, column=0, sticky="ew")
        ttk.Label(
            digit_card, textvariable=self.crops_heading, style="Section.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            digit_card,
            text="Prepared crops and model scores · scores are not calibrated probabilities.",
            style="Caption.TLabel",
            wraplength=620,
        ).pack(anchor="w", pady=(3, 8))
        self.digits_canvas = tk.Canvas(
            digit_card, bg="white", height=112, width=100, highlightthickness=0
        )
        self.digits_canvas.pack(fill="x")
        scroll = ttk.Scrollbar(
            digit_card, orient="horizontal", command=self.digits_canvas.xview
        )
        scroll.pack(fill="x")
        self.digits_canvas.configure(xscrollcommand=scroll.set)
        self.status_label = ttk.Label(
            root,
            textvariable=self.status,
            style="Muted.TLabel",
            wraplength=1120,
            padding=(16, 8),
        )
        self.status_label.pack(fill="x", side="bottom")
        root.bind(
            "<Configure>",
            lambda e: (
                self.status_label.configure(
                    wraplength=max(200, root.winfo_width() - 32)
                )
                if e.widget is root
                else None
            ),
        )
        root.bind("<MouseWheel>", self.scroll_page)
        root.bind("<FocusIn>", lambda e: self.ensure_visible(e.widget))
        for combo in self.combos:
            combo.bind("<MouseWheel>", self.scroll_page)
        self.controls = [
            self.load_button,
            self.folder_button,
            self.generate_button,
            self.folder_entry,
            self.number_entry,
        ]
        self.refresh_controls()
        self.poll_id = root.after(80, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def resize_page(self, event=None):
        width = max(1020, self.viewport.winfo_width())
        height = max(860, self.viewport.winfo_height(), self.page.winfo_reqheight())
        self.viewport.itemconfigure(self.page_item, width=width, height=height)
        self.viewport.configure(scrollregion=(0, 0, width, height))

    def scroll_page(self, event):
        steps = (
            -int(event.delta / 120)
            if abs(event.delta) >= 120
            else (-1 if event.delta > 0 else 1)
        )
        if event.state & 1:
            self.viewport.xview_scroll(steps, "units")
        else:
            self.viewport.yview_scroll(steps, "units")
        return "break"

    def ensure_visible(self, widget):
        """Bring keyboard-focused controls into the scrollable viewport."""
        if not str(widget).startswith(str(self.page) + "."):
            return
        self.root.update_idletasks()
        bounds = self.viewport.bbox(self.page_item)
        if not bounds:
            return
        for axis, pos, size, extent, visible in (
            (
                "x",
                widget.winfo_rootx() - self.page.winfo_rootx(),
                widget.winfo_width(),
                bounds[2],
                self.viewport.winfo_width(),
            ),
            (
                "y",
                widget.winfo_rooty() - self.page.winfo_rooty(),
                widget.winfo_height(),
                bounds[3],
                self.viewport.winfo_height(),
            ),
        ):
            current = (
                self.viewport.canvasx(0) if axis == "x" else self.viewport.canvasy(0)
            )
            target = (
                pos - 12
                if pos < current
                else (
                    pos + size - visible + 12
                    if pos + size > current + visible
                    else current
                )
            )
            getattr(self.viewport, axis + "view_moveto")(
                max(0, target) / max(1, extent)
            )

    def refresh_controls(self):
        for widget in self.controls:
            widget.configure(state="disabled" if self.busy else "normal")
        for combo in self.combos:
            combo.configure(state="disabled" if self.busy else "readonly")
        if self.mode.get() == "Alphanumeric":
            self.combos[3].configure(state="disabled")
        # Generated images resize every tile to one height, so line geometry carries no case information.
        self.case_check.configure(
            state="normal" if self.case_rule_usable() and not self.busy else "disabled"
        )
        self.recognise_button.configure(
            state="normal" if self.image is not None and not self.busy else "disabled"
        )
        self.export_button.configure(
            state="normal" if self.result is not None and not self.busy else "disabled"
        )

    def clear_result(self):
        self.result = None
        self.settings = None
        self.prediction.set("No prediction yet")
        self.details.set("Select Recognise to process this image.")
        self.digits_canvas.delete("all")
        self.redraw()
        self.refresh_controls()

    def settings_changed(self, event=None):
        self.clear_result()
        self.status.set("Settings changed. Run recognition again.")

    def case_rule_usable(self):
        return self.mode.get() == "Alphanumeric" and not (
            self.origin and self.origin.get("route") == "generated"
        )

    def open_lab(self):
        if self.lab is not None:
            self.lab.window.deiconify()
            self.lab.window.lift()
            return
        from gui_lab import ModelLab

        self.lab = ModelLab(self)

    def use_custom_model(self, mode, name):
        """Select a Model-lab checkpoint, switching mode if needed."""
        if self.busy:
            return
        if self.mode.get() != mode:
            self.mode.set(mode)
            self.mode_changed()
        self.combos[1].configure(values=self.service.models_for(mode))
        self.model.set(name)
        self.settings_changed()
        self.status.set(
            f"Using Model-lab checkpoint {name}. Shipped models remain selectable."
        )

    def mode_changed(self, event=None):
        characters = self.mode.get() == "Alphanumeric"
        self.combos[1].configure(values=self.service.models_for(self.mode.get()))
        self.model.set(DEFAULT_MODELS[self.mode.get()])
        self.method.set("grouped_components" if characters else "components")
        self.combos[2].configure(
            values=(
                ["grouped_components", "projection"]
                if characters
                else ["components", "projection"]
            )
        )
        self.representation.set("otsu")
        self.folder.set(
            str(
                ROOT
                / (
                    "data/own_characters/development"
                    if characters
                    else "data/segmentation_photos"
                )
            )
        )
        self.number.set("Ab12" if characters else "007")
        self.source_hint.set(
            "Use one line of separate letters and digits on pale paper."
            if characters
            else "Use one line of dark digits on pale paper."
        )
        self.folder_hint.set(
            "Single-character photo folder"
            if characters
            else "Single-digit photo folder"
        )
        self.request_hint.set(
            "Text to create  ·  1–12 letters / digits"
            if characters
            else "Number to create  ·  1–12 digits"
        )
        self.result_heading.set(
            "RECOGNISED TEXT" if characters else "RECOGNISED NUMBER"
        )
        self.regions_heading.set(
            "Detected characters" if characters else "Detected digits"
        )
        self.crops_heading.set(
            "Character breakdown" if characters else "Digit breakdown"
        )
        self.recognise_button.configure(
            text="Recognise text" if characters else "Recognise number"
        )
        self.clear_result()
        self.status.set(
            "Alphanumeric mode: ResNet-18 selected. Case matters; inspect similar letters and digits."
            if characters
            else "Number mode: baseline CNN selected. Run recognition for this mode."
        )

    def set_image(self, image, origin):
        self.image = image
        self.origin = origin
        self.source_text.set(
            "Generated image | Requested: " + origin["requested_text"]
            if origin["route"] == "generated"
            else "Photo | " + Path(origin["file"]).name
        )
        self.clear_result()

    def start_job(self, kind, function):
        if self.busy:
            return
        self.busy = True
        self.status.set(
            "Generating image..."
            if kind == "generate"
            else (
                "Loading image..." if kind == "load" else "Recognising handwriting..."
            )
        )
        self.refresh_controls()

        def worker():
            try:
                self.messages.put((kind, function(), None))
            except Exception as error:
                self.messages.put((kind, None, str(error)))

        self.job_thread = threading.Thread(target=worker, daemon=True)
        self.job_thread.start()

    def load_dialog(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a handwriting photo",
            initialdir=ROOT / "data",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        def load():
            return open_photo(path), {
                "route": "file",
                "file": str(Path(path).resolve()),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }

        self.clear_result()
        self.start_job("load", load)

    def folder_dialog(self):
        path = filedialog.askdirectory(
            parent=self.root,
            title="Folder of labelled single-character images",
            initialdir=ROOT / "data",
        )
        if path:
            self.folder.set(path)

    def generate(self):
        folder = self.folder.get()
        text = self.number.get().strip()
        generator = (
            generate_text if self.mode.get() == "Alphanumeric" else generate_number
        )
        self.clear_result()
        self.start_job("generate", lambda: generator(folder, text))

    def recognise(self):
        if self.image is None or self.busy:
            return
        image = self.image.copy()
        settings = {
            "mode": self.mode.get(),
            "model": self.model.get(),
            "method": self.method.get(),
            "representation": self.representation.get(),
            "line_case": self.line_case.get() and self.case_rule_usable(),
        }
        self.clear_result()

        def work():
            return self.service.recognise(image, settings)

        self.start_job("recognise", work)

    def poll(self):
        if self.closed:
            return
        try:
            kind, value, error = self.messages.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            if error:
                self.status.set("Could not complete: " + error)
                messagebox.showerror("Could not complete", error, parent=self.root)
            elif kind in ("generate", "load"):
                self.set_image(*value)
                warnings = value[1].get("warnings", [])
                self.status.set(
                    "Image ready. Select Recognise."
                    + (" Source warning: " + "; ".join(warnings) if warnings else "")
                )
            else:
                self.result, self.settings = value
                prediction = self.result["prediction"]
                seg = self.result["segmentation"]
                noun = (
                    "character" if self.settings["mode"] == "Alphanumeric" else "digit"
                )
                self.prediction.set(
                    prediction["text"] if prediction["text"] else f"No {noun}s detected"
                )
                details = f"{len(seg.boxes)} {noun}(s) | {self.settings['model'].upper()} | {self.settings['method']} | {self.settings['representation']}"
                case = prediction.get("case_rule")
                if case:
                    details += (
                        f" | line case: {len(case['changes'])} changed (classifier read {prediction['model_text']})"
                        if case["applied"]
                        else " | line case: no reference character, classifier case kept"
                    )
                self.details.set(details)
                self.status.set(
                    "; ".join(seg.warnings)
                    if seg.warnings
                    else "Recognition finished. Inspect the crops and compare the result with the handwriting."
                )
                self.redraw()
                self.draw_digits()
            self.refresh_controls()
        self.poll_id = self.root.after(80, self.poll)

    def redraw(self):
        if not hasattr(self, "output_canvas"):
            return
        self.photos = []
        detected = annotated_image(self.result["segmentation"]) if self.result else None
        for canvas, image, placeholder in [
            (self.input_canvas, self.image, "Open or generate an image"),
            (
                self.output_canvas,
                detected,
                "Character regions appear after recognition",
            ),
        ]:
            canvas.delete("all")
            w = max(50, canvas.winfo_width())
            h = max(50, canvas.winfo_height())
            if image is None:
                canvas.create_text(
                    w / 2,
                    h / 2,
                    text=placeholder,
                    fill="#8493a8",
                    font=("Segoe UI", 10),
                    width=w - 40,
                )
            else:
                image = image.copy()
                image.thumbnail((max(1, w - 20), max(1, h - 20)))
                photo = tk_image(image, self.root)
                self.photos.append(photo)
                canvas.create_image(w / 2, h / 2, image=photo)

    def draw_digits(self):
        self.digit_photos = []
        self.digits_canvas.delete("all")
        if not self.result:
            return
        prediction = self.result["prediction"]
        items = prediction.get("characters", prediction.get("digits", []))
        for index, (array, item) in enumerate(zip(self.result["prepared"], items)):
            image = Image.fromarray(np.rint(array * 255).astype("uint8")).resize(
                (64, 64), Image.Resampling.NEAREST
            )
            photo = tk_image(image, self.root)
            self.digit_photos.append(photo)
            x = index * 128 + 8
            self.digits_canvas.create_rectangle(
                x, 0, x + 118, 106, fill="#f3f6fb", outline=""
            )
            self.digits_canvas.create_image(x + 27, 8, image=photo, anchor="nw")
            symbol = item.get("character", item.get("digit"))
            if "model_character" in item:
                symbol = f"{symbol}←{item['model_character']}"  # final←classifier
            self.digits_canvas.create_text(
                x + 59,
                87,
                anchor="center",
                text=f"{symbol}   ·   {item['confidence']:.1%}",
                fill="#334155",
                font=("Segoe UI", 10, "bold"),
            )
        self.digits_canvas.configure(
            scrollregion=(0, 0, max(1, len(self.digit_photos)) * 128, 110)
        )

    def export_dialog(self):
        if not self.result or self.busy:
            return
        folder = filedialog.askdirectory(
            parent=self.root,
            title="Save a new evidence folder here",
            initialdir=ROOT / "evidence",
        )
        if not folder:
            return
        try:
            destination = export_result(
                folder, self.image, self.origin, self.result, self.settings
            )
            self.status.set("Saved evidence: " + str(destination))
        except (OSError, ValueError) as error:
            messagebox.showerror("Export failed", str(error), parent=self.root)

    def close(self):
        self.closed = True
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def main():
    root = tk.Tk()
    HNRSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
