"""Small Tk canvas charts for the Model lab: lines, bars and a confusion heatmap.

Pure Tkinter so the GUI needs no plotting dependency. Each function clears the
canvas and draws to its current size.
"""

from tkinter import font as tkfont

INK = "#192b46"
MUTED = "#64748b"
GRID = "#e2e8f0"
SERIES = (
    "#2563eb",
    "#d97706",
    "#059669",
    "#db2777",
    "#7c3aed",
    "#0891b2",
    "#65a30d",
    "#dc2626",
    "#475569",
)
FONT = ("Segoe UI", 9)


def _size(canvas):
    canvas.update_idletasks()
    return max(canvas.winfo_width(), int(canvas["width"])), max(
        canvas.winfo_height(), int(canvas["height"])
    )


def _placeholder(canvas, message):
    width, height = _size(canvas)
    canvas.create_text(
        width / 2,
        height / 2,
        text=message,
        fill=MUTED,
        font=("Segoe UI", 10),
        width=width - 40,
    )


def line_chart(canvas, series, title, y_label="", y_range=None, x_label="Epoch"):
    """series: {name: [(x, y), ...]}. Points are joined in the given order."""
    canvas.delete("all")
    points = [p for values in series.values() for p in values]
    if not points:
        return _placeholder(canvas, f"{title}: no data yet")
    width, height = _size(canvas)
    left, right, top, bottom = 52, width - 16, 30, height - 64
    xs = [p[0] for p in points]
    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        x0, x1 = x0 - 1, x1 + 1
    y0, y1 = y_range or (min(p[1] for p in points), max(p[1] for p in points))
    if y1 == y0:
        y0, y1 = y0 - 0.5, y1 + 0.5
    sx = lambda x: left + (x - x0) / (x1 - x0) * (right - left)
    sy = lambda y: bottom - (y - y0) / (y1 - y0) * (bottom - top)
    canvas.create_text(
        left, 12, text=title, anchor="w", fill=INK, font=("Segoe UI", 10, "bold")
    )
    for step in range(5):
        value = y0 + (y1 - y0) * step / 4
        canvas.create_line(left, sy(value), right, sy(value), fill=GRID)
        canvas.create_text(
            left - 6, sy(value), text=f"{value:.3g}", anchor="e", fill=MUTED, font=FONT
        )
    whole = all(float(x).is_integer() for x in xs)
    ticks = (
        sorted(set(xs))
        if whole and len(set(xs)) <= 12
        else [x0 + (x1 - x0) * i / 4 for i in range(5)]
    )
    for x in ticks:
        canvas.create_text(
            sx(x),
            bottom + 12,
            text=f"{x:g}" if whole else f"{x:.1f}",
            fill=MUTED,
            font=FONT,
        )
    canvas.create_text(
        (left + right) / 2, bottom + 27, text=x_label, fill=MUTED, font=FONT
    )
    if y_label:
        canvas.create_text(
            8, (top + bottom) / 2, text=y_label, angle=90, fill=MUTED, font=FONT
        )
    legend_x = left
    for index, (name, values) in enumerate(series.items()):
        colour = SERIES[index % len(SERIES)]
        coordinates = [c for x, y in values for c in (sx(x), sy(y))]
        if len(values) > 1:
            canvas.create_line(*coordinates, fill=colour, width=2, smooth=False)
        for x, y in values if len(values) <= 30 else values[-1:]:
            canvas.create_oval(
                sx(x) - 3, sy(y) - 3, sx(x) + 3, sy(y) + 3, fill=colour, outline=""
            )
        canvas.create_rectangle(
            legend_x, height - 14, legend_x + 10, height - 4, fill=colour, outline=""
        )
        label = canvas.create_text(
            legend_x + 14, height - 9, text=name, anchor="w", fill=INK, font=FONT
        )
        legend_x = canvas.bbox(label)[2] + 14


def bar_chart(
    canvas, labels, values, title, value_format="{:.2%}", highlight=None, y_range=None
):
    """Horizontal bars, one per label; `highlight` names bars to emphasise."""
    canvas.delete("all")
    if not values:
        return _placeholder(canvas, f"{title}: no data")
    width, height = _size(canvas)
    canvas.create_text(
        12, 12, text=title, anchor="w", fill=INK, font=("Segoe UI", 10, "bold")
    )
    lo, hi = y_range or (min(0, min(values)), max(values))
    measure = tkfont.Font(root=canvas, font=FONT).measure
    label_width = min(width * 0.45, max(measure(str(label)) for label in labels) + 18)
    row = (height - 34) / len(values)
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 30 + index * row
        length = (value - lo) / (hi - lo) * (width - label_width - 80) if hi > lo else 0
        colour = SERIES[0] if not highlight or label in highlight else "#93c5fd"
        canvas.create_text(
            label_width - 8, y + row / 2, text=label, anchor="e", fill=INK, font=FONT
        )
        canvas.create_rectangle(
            label_width,
            y + row * 0.18,
            label_width + max(1, length),
            y + row * 0.82,
            fill=colour,
            outline="",
        )
        canvas.create_text(
            label_width + max(1, length) + 6,
            y + row / 2,
            text=value_format.format(value),
            anchor="w",
            fill=INK,
            font=FONT,
        )


def heatmap(canvas, matrix, classes, title, show_counts=None):
    """Confusion matrix: rows are true classes, columns predicted classes.

    Cells are shaded by row-normalised rate so rare classes stay visible;
    counts are printed when the matrix is small enough to read.
    """
    canvas.delete("all")
    if not matrix:
        return _placeholder(canvas, f"{title}: no data")
    width, height = _size(canvas)
    size = len(matrix)
    show_counts = size <= 12 if show_counts is None else show_counts
    margin_left, margin_top = 36, 44
    cell = max(
        2.0, min((width - margin_left - 12) / size, (height - margin_top - 12) / size)
    )
    canvas.create_text(
        12, 12, text=title, anchor="w", fill=INK, font=("Segoe UI", 10, "bold")
    )
    canvas.create_text(
        margin_left + cell * size / 2, 30, text="Predicted", fill=MUTED, font=FONT
    )
    label_font = ("Segoe UI", 8 if size <= 12 else 6)
    step = 1 if cell >= 9 else 2
    for index, name in enumerate(classes):
        if index % step == 0:
            canvas.create_text(
                margin_left + cell * (index + 0.5),
                margin_top - 6,
                text=name,
                fill=MUTED,
                font=label_font,
            )
            canvas.create_text(
                margin_left - 6,
                margin_top + cell * (index + 0.5),
                text=name,
                anchor="e",
                fill=MUTED,
                font=label_font,
            )
    for r, values in enumerate(matrix):
        total = sum(values) or 1
        for c, value in enumerate(values):
            rate = value / total
            if r == c:
                shade = (
                    int(236 - 200 * rate),
                    int(253 - 120 * rate),
                    int(245 - 170 * rate),
                )
            else:
                # Errors: white to red, made visible even at low rates.
                strength = min(1.0, rate * 4)
                shade = (
                    int(255 - 35 * strength),
                    int(255 - 200 * strength),
                    int(255 - 200 * strength),
                )
            x, y = margin_left + c * cell, margin_top + r * cell
            canvas.create_rectangle(
                x,
                y,
                x + cell,
                y + cell,
                fill="#%02x%02x%02x" % shade,
                outline="white" if cell > 6 else "",
            )
            if show_counts and value:
                canvas.create_text(
                    x + cell / 2,
                    y + cell / 2,
                    text=str(value),
                    fill="white" if r == c and rate > 0.6 else INK,
                    font=("Segoe UI", 8),
                )
