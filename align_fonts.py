"""
Script to calculate offsets for the characters of a font in order to align them (pixel-wise) to another font.
"""

import itertools, json, base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


DEFAULT_CHARS = [chr(c) for c in itertools.chain(range(ord("a"), ord("z") + 1), range(ord("A"), ord("Z") + 1))]
OUTPUT_FOLDER = "out"


def align_font(overlay_font_path, base_font_path, font_size=128, score_epsilon=1.0, scale_epsilon=0.01, resolution=25) -> dict:
    base_font = ImageFont.truetype(base_font_path, font_size)
    low_scale, high_scale = 0.5, 2.0
    while True:
        scales = np.linspace(low_scale, high_scale, resolution)
        results = [align_font_instance(ImageFont.truetype(overlay_font_path, scale * font_size), base_font)
                   for scale in scales]
        min_i = np.argmin([r["total_overlap"] for r in results])
        assert 0 < min_i < len(results) - 1, "Initial bounds are too narrow."
        low_scale, high_scale = scales[min_i - 1], scales[min_i + 1]

        if (
            (abs(results[min_i - 1]["total_overlap"] - results[min_i]["total_overlap"]) < score_epsilon and
             abs(results[min_i + 1]["total_overlap"] - results[min_i]["total_overlap"]) < score_epsilon) or
            high_scale - low_scale < scale_epsilon
        ):
            return {
                "base_font": Path(base_font_path).stem,
                "base_font_path": base_font_path,
                "overlay_font": Path(overlay_font_path).stem,
                "overlay_font_path": overlay_font_path,
                "font_scale": scales[min_i],
                **results[min_i]
            }


def align_font_instance(overlay_font: ImageFont.FreeTypeFont, base_font: ImageFont.FreeTypeFont, charset=DEFAULT_CHARS) -> dict:
    """
    Takes two font objects and returns a dictionary with optimal offsets, individual scores, and a total score.
    """
    optimization_result = {"total_overlap": None, "offsets": {}}
    for char in charset:
        optimization_result["offsets"][char] = optimize_offset(char, overlay_font, base_font)
    optimization_result["total_overlap"] = sum([v[1] for v in optimization_result["offsets"].values()])
    return optimization_result


def optimize_offset(char, overlay_font, ref_font):
    ref_img = char_to_image_array(char, ref_font) / 255.0
    overlay_img = char_to_image_array(char, overlay_font, (ref_img.shape[1], ref_img.shape[0])) / 255.0
    offset = (0, 0)
    overlaps = {offset: np.sum(overlay_img * (1.0 - ref_img))}
    neighbors = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    while True:
        neighbor_overlaps = []
        for dx, dy in neighbors:
            new_offset = (offset[0] + dx, offset[1] + dy)
            if new_offset not in overlaps:
                img = move_image(overlay_img, *new_offset)
                overlap = np.sum(img * (1.0 - ref_img))
                overlaps[new_offset] = overlap
            neighbor_overlaps.append(overlaps[new_offset])
        i = np.argmin(neighbor_overlaps)
        if neighbor_overlaps[i] >= overlaps[offset]:
            break
        offset = tuple(np.asarray(offset) + neighbors[i])
    score = overlaps[offset]
    offset = tuple(np.asarray(offset) / overlay_font.size)
    return offset, score


def move_image(img, dx, dy):
    if dx > 0:
        img = np.pad(img, ((0, 0), (dx, 0)), mode="constant", constant_values=255)[:, :-dx]
    if dx < 0:
        img = np.pad(img, ((0, 0), (0, -dx)), mode="constant", constant_values=255)[:, -dx:]
    if dy > 0:
        img = np.pad(img, ((dy, 0), (0, 0)), mode="constant", constant_values=255)[:-dy, :]
    if dy < 0:
        img = np.pad(img, ((0, -dy), (0, 0)), mode="constant", constant_values=255)[-dy:, :]
    return img


def char_to_image_array(char, font, image_size=None):
    # Create a blank image with white background
    if image_size is None:
        image_size = (int(2 * font.size), int(2 * font.size))
    image = Image.new("L", image_size, 255)
    draw = ImageDraw.Draw(image)

    # Render the text
    # bb = draw.textbbox((0, 0), char, font=font)
    draw.text((0.125 * image_size[0], 0.667 * image_size[0]), char, font=font, anchor="ls", fill=0)

    # Convert image to NumPy array
    image_array = np.array(image, float)

    return image_array


def write_json(align_font_result):
    filename = f"{align_font_result['overlay_font']}_on_{align_font_result['base_font']}.json"
    with open(f"{OUTPUT_FOLDER}/{filename}", "w") as f:
        json.dump(align_font_result, f, indent=2)


def write_report(align_font_result, charset=DEFAULT_CHARS, font_size=128):
    filename = f"{align_font_result['overlay_font']}_on_{align_font_result['base_font']}.html"
    base_font = ImageFont.truetype(align_font_result["base_font_path"], font_size)
    overlay_font = ImageFont.truetype(align_font_result["overlay_font_path"], align_font_result['font_scale'] * font_size)
    with open(f"{OUTPUT_FOLDER}/{filename}", "w") as f:
        f.write("<html><body>")
        f.write(f"<h1>{align_font_result['overlay_font']} on {align_font_result['base_font']}</h1>")
        f.write(f"<h2>Font scale: {align_font_result['font_scale']}</h2>")
        f.write(f"<h2>Total overlap: {align_font_result['total_overlap']}</h2>")
        f.write("<table>")
        f.write("<tr><th>Char</th><th>Offset</th><th>Score</th><th>Image</th></tr>")
        for char in charset:
            offset, score = align_font_result['offsets'][char]
            f.write(f"<tr><td>{char}</td><td>{offset}</td><td>{score}</td>")
            img = draw_char_overlay(char, overlay_font, base_font, offset)
            img_base64 = base64.b64encode(cv2.imencode(".png", img)[1]).decode("utf-8")
            f.write(f"<td><img src='data:image/png;base64,{img_base64}'></td></tr>")
        f.write("</table>")
        f.write("</body></html>")


def draw_char_overlay(char: str, overlay_font, base_font, offset):
    offset = (int(offset[0] * overlay_font.size), int(offset[1] * overlay_font.size))
    r = 255.0 - char_to_image_array(char, base_font)
    g = 255.0 - move_image(char_to_image_array(char, overlay_font, (r.shape[1], r.shape[0])), *offset)
    b = np.zeros_like(r)
    overlap = np.minimum(g, r) > 0
    b[overlap] = 0.5 * (r[overlap] + g[overlap])
    img = np.stack([b, g, r], axis=2)
    return img


if __name__ == "__main__":
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    print("Optimizing...", flush=True)
    result = align_font("fonts/LiberationSans-BoldItalic.ttf", "trash/fonts/Arial-Italic.ttf")
    print("Writing json...", flush=True)
    write_json(result)

    # with open("Merriweather-Bold_on_TimesNewRoman.json") as f:
    #     result = json.load(f)

    print("Writing report...", flush=True)
    write_report(result)


# print(align_font_instance(ImageFont.truetype("fonts/LibreBaskerville-Bold.ttf", 0.858 * font_size), ImageFont.truetype("fonts/TimesNewRoman.ttf", font_size)))

# for alpha in np.linspace(0.9310246913580247, 0.9383333333333334, 10):
#     font = ImageFont.truetype("fonts/Merriweather-Bold.ttf", alpha * font_size)

#     result = align_font(font, ref_font)
#     total_overlap = sum(o for _, o in result.values())
#     print(alpha, total_overlap)




# img = char_to_image_array('y', font)
# img = move_image(img, -10, 0)

# align_font(font, ref_font)


# char = 'W'

# offset, score = optimize_offset(char, font, ref_font)
# print(offset, score)

# offset = (int(offset[0] * font.size), int(offset[1] * font.size))

# r = 255.0 - move_image(char_to_image_array(char, font), *offset)
# g = 255.0 - char_to_image_array(char, ref_font, (r.shape[1], r.shape[0]))
# b = np.zeros_like(r)
# img = np.stack([b, g, r], axis=2)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

