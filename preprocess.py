from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "own_images" / "Paarth_01_label-0.png"
OUTPUT = ROOT / "evidence" / "preprocessing"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Open the photo and correct its camera orientation.
with Image.open(INPUT) as image:
    original = ImageOps.exif_transpose(image).convert("RGB")

# Convert colour pixels into shades of grey.
grayscale = ImageOps.grayscale(original)
grayscale.save(OUTPUT / "single_digit_grayscale.png")

# Make smaller previews for a side-by-side comparison.
original_preview = original.copy()
original_preview.thumbnail((400, 400))

gray_preview = grayscale.convert("RGB")
gray_preview.thumbnail((400, 400))

comparison = Image.new("RGB", (820, 450), "white")
comparison.paste(original_preview, (10, 40))
comparison.paste(gray_preview, (420, 40))

draw = ImageDraw.Draw(comparison)
draw.text((10, 10), "Original", fill="black")
draw.text((420, 10), "Grayscale", fill="black")

comparison.save(OUTPUT / "original_vs_grayscale.png")

print("Original image size:", original.size)
print("Grayscale image mode:", grayscale.mode)
print("Saved outputs to:", OUTPUT)