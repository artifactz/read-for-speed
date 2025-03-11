"""
Script to calculate offsets for the characters of a font in order to align them (pixel-wise) to another font.
"""

import json, base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from tqdm import tqdm


DEFAULT_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
OUTPUT_FOLDER = "out"
REMAP_FOLDER = "remap"


def run_pool(font_map: dict, install_remap=False, verbose=False):
    import multiprocessing
    pool = multiprocessing.Pool()
    args_list = [(overlay_font_path, base_font_path, install_remap, verbose)
                 for overlay_font_path, base_font_path in font_map.items()]
    list(tqdm(
        pool.imap_unordered(run_wrapper, args_list),
        total=len(font_map)
    ))


def run(overlay_font_path, base_font_path, install_remap=False, verbose=True):
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    if verbose:
        print("Optimizing", Path(overlay_font_path).stem, "on", Path(base_font_path).stem, "...", flush=True)
    result = align_font(overlay_font_path, base_font_path)
    if verbose:
        print("Writing json...", flush=True)
    write_json(result)
    if install_remap:
        write_json(result, f"{REMAP_FOLDER}/{Path(base_font_path).stem}.json")

    if verbose:
        print("Writing report...", flush=True)
    write_report(result)


def align_font(overlay_font_path, base_font_path, font_size=128, score_epsilon=1e-5, scale_epsilon=0.01, resolution=25) -> dict:
    base_font = ImageFont.truetype(base_font_path, font_size)
    low_scale, high_scale = 0.5, 2.0
    while True:
        scales = np.linspace(low_scale, high_scale, resolution)
        results = [align_font_instance(ImageFont.truetype(overlay_font_path, scale * font_size), base_font)
                   for scale in scales]
        min_i = np.argmin([r["average_remainder"] for r in results])
        assert 0 < min_i < len(results) - 1, "Initial bounds are too narrow."
        low_scale, high_scale = scales[min_i - 1], scales[min_i + 1]

        if (
            (abs(results[min_i - 1]["average_remainder"] - results[min_i]["average_remainder"]) < score_epsilon and
             abs(results[min_i + 1]["average_remainder"] - results[min_i]["average_remainder"]) < score_epsilon) or
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
    optimization_result = {"characters": {}, "average_remainder": None}
    for char in charset:
        optimization_result["characters"][char] = optimize_offset(char, overlay_font, base_font)
    optimization_result["average_remainder"] = np.mean(
        [value["remainder"] for value in optimization_result["characters"].values()]
    )
    optimization_result["median_y_offset"] = sorted(
        [value["offset"][1] for value in optimization_result["characters"].values()]
    )[len(optimization_result["characters"]) // 2]

    return optimization_result


def optimize_offset(char, overlay_font, ref_font):
    """
    Gradient-descents the character offset by one pixel (incl. diagonals) until a local minimum remainder is reached.
    The remainder is the number of pixels of the underlaying character not covered by the overlaying character.
    Returns offset and ratio of remainder pixels.
    """
    # Start with both characters centered horizontally
    ref_img = create_char_image(char, ref_font)
    overlay_img = create_char_image(char, overlay_font, (ref_img["image"].shape[1], ref_img["image"].shape[0]))
    ref_img["image"] /= 255.0
    overlay_img["image"] /= 255.0
    offset = (0, 0)
    remainders = {offset: np.sum(overlay_img["image"] * (1.0 - ref_img["image"]))}
    neighbors = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    while True:
        neighbor_remainders = []
        for dx, dy in neighbors:
            new_offset = (offset[0] + dx, offset[1] + dy)
            if new_offset not in remainders:
                img = move_image(overlay_img["image"], *new_offset)
                remainder = np.sum(img * (1.0 - ref_img["image"]))
                remainders[new_offset] = remainder
            neighbor_remainders.append(remainders[new_offset])
        i = np.argmin(neighbor_remainders)
        if neighbor_remainders[i] >= remainders[offset]:
            break
        offset = tuple(np.asarray(offset) + neighbors[i])

    remainder = remainders[offset] / (ref_img["image"].shape[0] * ref_img["image"].shape[1])  # absolute number of pixels to ratio
    initial_x_offset = overlay_img["bbox"][0] - ref_img["bbox"][0]  # from horizontal centering
    offset = (offset[0] + initial_x_offset, offset[1])
    offset = tuple(np.asarray(offset) / overlay_font.size)
    size = ((overlay_img["bbox"][2] - overlay_img["bbox"][0]) / overlay_font.size,
            (overlay_img["bbox"][3] - overlay_img["bbox"][1]) / overlay_font.size)

    return {
        "size": size,
        "offset": offset,
        "remainder": remainder,
    }


def move_image(img, dx, dy, pad_color=255):
    if dx > 0:
        img = np.pad(img, ((0, 0), (dx, 0)), mode="constant", constant_values=pad_color)[:, :-dx]
    if dx < 0:
        img = np.pad(img, ((0, 0), (0, -dx)), mode="constant", constant_values=pad_color)[:, -dx:]
    if dy > 0:
        img = np.pad(img, ((dy, 0), (0, 0)), mode="constant", constant_values=pad_color)[:-dy, :]
    if dy < 0:
        img = np.pad(img, ((0, -dy), (0, 0)), mode="constant", constant_values=pad_color)[-dy:, :]
    return img


def create_char_image(char, font, image_size=None, x=None):
    """
    Creates an image of the character with the given font and size.
    """
    if image_size is None:
        image_size = (int(1.333 * font.size), int(1.333 * font.size))
    # Create a blank image with white background
    image = Image.new("L", image_size, 255)
    draw = ImageDraw.Draw(image)
    bb = draw.textbbox((0, 0), char, font=font, anchor="ls")

    # Determine position
    y = 0.75 * image_size[0]
    if x is None:
        w = bb[2] - bb[0] + 1
        x = int((image_size[0] - w) / 2)

    # Render the text
    draw.text((x, y), char, font=font, anchor="ls", fill=0)

    # Convert image to NumPy array
    image_array = np.array(image, float)

    return {
        "image": image_array,
        "bbox": (bb[0] + x, bb[1] + y, bb[2] + x, bb[3] + y)
    }


def write_json(align_font_result, path=None):
    if not path:
        path = f"{OUTPUT_FOLDER}/{align_font_result['overlay_font']}_on_{align_font_result['base_font']}.json"
    align_font_result = dict(align_font_result)
    del align_font_result["base_font_path"]
    del align_font_result["overlay_font_path"]
    with open(path, "w") as f:
        json.dump(align_font_result, f, indent=2)


def write_report(align_font_result, charset=DEFAULT_CHARS, font_size=128):
    filename = f"{align_font_result['overlay_font']}_on_{align_font_result['base_font']}.html"
    base_font = ImageFont.truetype(align_font_result["base_font_path"], font_size)
    overlay_font = ImageFont.truetype(align_font_result["overlay_font_path"], align_font_result['font_scale'] * font_size)

    with open(f"{OUTPUT_FOLDER}/{filename}", "w") as f:
        f.write("<html><head><style>"
                "table, th, td {"
                "  border-collapse: collapse;"
                "  border: 1px solid #ccc;"
                "}"
                "th, td {"
                "  padding: 5px;"
                "  text-align: center;"
                "}"
                "</style></head><body>")
        f.write(f"<h1>{align_font_result['overlay_font']} on {align_font_result['base_font']}</h1>")
        f.write(f"<h2>Font scale: {align_font_result['font_scale']}</h2>")
        f.write(f"<h2>Average remainder: {align_font_result['average_remainder'] * 100:.5f}%</h2>")
        f.write("<table>")
        f.write("<tr><th>Char</th><th>Offset</th><th>Remainder</th><th>Image</th></tr>")

        for char in charset:
            c = align_font_result['characters'][char]
            offset, remainder = c["offset"], c["remainder"]
            f.write(f"<tr><td>{char}</td><td>({offset[0]:.3f}, {offset[1]:.3f})</td><td>{remainder * 100:.5f}%</td>")
            img = draw_char_overlay(char, overlay_font, base_font, offset)
            img_base64 = base64.b64encode(cv2.imencode(".png", img)[1]).decode("utf-8")
            f.write(f"<td><img src='data:image/png;base64,{img_base64}'></td></tr>")

        f.write("</table>")
        f.write("</body></html>")


def draw_char_overlay(char: str, overlay_font, base_font, offset):
    """
    Draws both the base and overlay version of the character and highlights remainder pixels.
    """
    offset = (int(offset[0] * overlay_font.size), int(offset[1] * overlay_font.size))
    # img1, x = create_char_image(char, base_font)
    char1_image = create_char_image(char, base_font)
    w, h, x = char1_image["image"].shape[1], char1_image["image"].shape[0], char1_image["bbox"][0]
    char2_image = create_char_image(char, overlay_font, (w, h), x)
    r = 255.0 - char1_image["image"]
    g = 255.0 - move_image(char2_image["image"], *offset)
    b = np.zeros_like(r)
    overlap = np.minimum(g, r) > 0
    b[overlap] = 0.5 * (r[overlap] + g[overlap])
    img = np.stack([b, g, r], axis=2)
    return img


def run_wrapper(arg):
    run(*arg)


def regenerate_remappings():
    """
    Re-runs alignment for the current selection of open fonts as substitutes to their corresponding proprietary font.
    Copies the result to the remap/ folder to be used by pdf_overlay.
    """
    font_map = {
        "fonts/Junicode-BoldItalic.ttf": "proprietary/fonts/AGaramond-Italic.otf",
        "fonts/EBGaramond-Bold.ttf": "proprietary/fonts/AGaramond.otf",
        "fonts/IBMPlexSans-Bold.ttf": "proprietary/fonts/Aptos.ttf",
        "fonts/UnBPro-Extrabold.ttf": "proprietary/fonts/Arial-Bold.ttf",
        "fonts/LiberationSans-BoldItalic.ttf": "proprietary/fonts/Arial-Italic.ttf",
        "fonts/LiberationSans-Bold.ttf": "proprietary/fonts/Arial.ttf",
        "fonts/ComputerModernSerif-BoldItalic.ttf": "fonts/ComputerModernSerif-Italic.ttf",
        "fonts/ComputerModernSerif-Bold.ttf": "fonts/ComputerModernSerif.ttf",
        "fonts/Vegur-Bold.ttf": "proprietary/fonts/Corbel.ttf",
        "fonts/LinBiolinum-BoldItalic.ttf": "proprietary/fonts/LinBiolinum-Italic.ttf",
        "fonts/Mignon-BoldItalic.ttf": "proprietary/fonts/MinionPro-Italic.ttf",
        "fonts/Mignon-Bold.ttf": "proprietary/fonts/MinionPro.ttf",
        "fonts/STIXTwoText-Bold.ttf": "proprietary/fonts/STIXTwoText.ttf",
        "fonts/CrimsonText-BoldItalic.ttf": "proprietary/fonts/Sabon-Italic.ttf",
        "fonts/CrimsonText-Bold.ttf": "proprietary/fonts/Sabon.ttf",
        "fonts/Playfair_SemiCondensed-Bold.ttf": "proprietary/fonts/SuisseWorks.otf",
        "fonts/TimesNewerRoman-BoldItalic.ttf": "proprietary/fonts/TimesNewRoman-Italic.ttf",
        "fonts/TimesNewerRoman-Bold.ttf": "proprietary/fonts/TimesNewRoman.ttf",
        "fonts/OpenSans_SemiCondensed-SemiBold.ttf": "proprietary/fonts/VectoraLH-Light.ttf",
        "fonts/OpenSans_SemiCondensed-Bold.ttf": "proprietary/fonts/VectoraLH.ttf",
        "fonts/DejaVuSans-BoldItalic.ttf": "proprietary/fonts/Verdana-Italic.ttf",
        "fonts/DejaVuSans-Bold.ttf": "proprietary/fonts/Verdana.ttf",
    }
    run_pool(font_map, install_remap=True)


if __name__ == "__main__":
    # regenerate_remappings()

    run_pool({
        "proprietary/download/urw-core35-fonts-master/C059-Bold.ttf": "proprietary/fonts/URWPalladioL.otf",
        "proprietary/download/urw-core35-fonts-master/NimbusMonoPS-Bold.ttf": "proprietary/fonts/URWPalladioL.otf",
        "proprietary/download/urw-core35-fonts-master/NimbusRoman-Bold.ttf": "proprietary/fonts/URWPalladioL.otf",
        "proprietary/download/urw-core35-fonts-master/NimbusSans-Bold.ttf": "proprietary/fonts/URWPalladioL.otf",
        "proprietary/download/urw-core35-fonts-master/NimbusSansNarrow-Bold.ttf": "proprietary/fonts/URWPalladioL.otf",
        "proprietary/download/urw-core35-fonts-master/P052-Bold.ttf": "proprietary/fonts/URWPalladioL.otf",
        "proprietary/download/urw-core35-fonts-master/URWBookman-Demi.ttf": "proprietary/fonts/URWPalladioL.otf",
        "proprietary/download/urw-core35-fonts-master/URWBookman-Light.ttf": "proprietary/fonts/URWPalladioL.otf",
        "proprietary/download/urw-core35-fonts-master/URWGothic-Book.ttf": "proprietary/fonts/URWPalladioL.otf",
        "proprietary/download/urw-core35-fonts-master/URWGothic-Demi.ttf": "proprietary/fonts/URWPalladioL.otf",
    })

    run("proprietary/download/urw-core35-fonts-master/ttf/P052.ttf", "proprietary/fonts/URWPalladioL.otf")
