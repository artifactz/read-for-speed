"""
Script to generate data/font_extents.csv containing ascent and descent ratios for all fonts found in `folders`.
These ratios are later used to determine bounding boxes, see `fonts.get_font_extents`.
"""

import itertools
from PIL import ImageFont
from pathlib import Path


folders = ["fonts", "proprietary/fonts"]
output_csv_path = "data/font_extents.csv"
resolution = 1000


def run():
    paths = [path for folder in folders for path in itertools.chain(Path(folder).glob("*.ttf"), Path(folder).glob("*.otf"))]
    paths = sorted(set([p for p in paths if p.stem == path.stem][0] for path in paths), key=lambda p: p.stem.lower())

    csv = "fontname,ascent,descent\n"
    for path in paths:
        font = ImageFont.truetype(path, resolution)
        box = font.getbbox("[QlIgp]", anchor="ls")
        ascent, descent = -box[1] / resolution, box[3] / resolution
        csv += f"{path.stem},{ascent},{descent}\n"

    with open(output_csv_path, "w", encoding="utf-8") as f:
        f.write(csv)


if __name__ == "__main__":
    run()

    import csv
    with open(output_csv_path) as f:
        result = {d['fontname']: {k: v for k, v in d.items() if k != 'fontname'} for d in csv.DictReader(f)}
        print(result)
