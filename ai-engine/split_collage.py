"""
split_collage.py — Splits an irregular grid-collage (3 / 3 / 4 tiles per row)
into 10 separate screenshot files.
"""

from PIL import Image
from pathlib import Path

# --- EDIT THESE TO MATCH YOUR FILES ---
INPUT_PATH = r"D:\MM_Test\ChatGPT Image Jul 8, 2026, 08_12_44 AM.png"
OUTPUT_DIR = r"D:\MM_Test\split_screenshots"
ROW_LAYOUT = [3, 3, 4]   # number of tiles in each row, top to bottom
# ----------------------------------------

img = Image.open(INPUT_PATH)
width, height = img.size

num_rows = len(ROW_LAYOUT)
row_height = height // num_rows

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

count = 0
for row_index, cols_in_this_row in enumerate(ROW_LAYOUT):
    top = row_index * row_height
    bottom = top + row_height if row_index < num_rows - 1 else height  # last row absorbs rounding
    col_width = width // cols_in_this_row

    for col_index in range(cols_in_this_row):
        left = col_index * col_width
        right = left + col_width if col_index < cols_in_this_row - 1 else width  # last col absorbs rounding
        tile = img.crop((left, top, right, bottom))
        count += 1
        out_path = Path(OUTPUT_DIR) / f"screenshot_{count:02d}.png"
        tile.save(out_path)
        print(f"Saved {out_path}  (row {row_index+1}, col {col_index+1})")

print(f"Done — {count} tiles saved to {OUTPUT_DIR}")