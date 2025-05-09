import math, re, tempfile, collections, subprocess, logging
from typing import IO
from tqdm import tqdm
import pdfplumber
from reportlab.pdfgen.canvas import Canvas
import fonts
from ml.font import extract
from util.memory_usage import get_memory_usage_mb, get_top_memory_users_mb


DEFAULT_CONFIG = {
    "draw_bbox": True,  # Hides existing characters by drawing a filled bounding box.
    "typesetting_mode": "x_offset",  # How to position characters: "full_offset", "x_offset", or "rearranged".
    "use_extrabold": False  # Will overlay bold text with extrabold text. Skipped otherwise.
}


# fallback characters to use for looking up offsets
CHAR_MAP = {
    "Ä": "A",
    "Ö": "O",
    "Ü": "U",
    "ä": "a",
    "ö": "o",
    "ü": "u",
    "Á": "A",
    "É": "E",
    "Í": "I",
    "Ó": "O",
    "Ú": "U",
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "À": "A",
    "È": "E",
    "Ì": "I",
    "Ò": "O",
    "Ù": "U",
    "à": "a",
    "è": "e",
    "ì": "i",
    "ò": "o",
    "ù": "u",
}


def _is_encrypted_font(font_name: str) -> bool:
    return re.search(r"Adv[A-Z]", font_name) is not None


def _split_emphasized_part(word_chars):
    if len(word_chars) < 2:
        return [], word_chars

    word = "".join(char["text"] for char in word_chars)

    if (
        re.match(r"^\d+(?:\w|\w\w)?$", word) or  # Word consists of digits, optionally followed by up to 2 letters
        word == word.upper() or  # Word is upper case
        (len(word) > 2 and word[-1] == "s" and word[:-1] == word[:-1].upper())  # Word is upper case/plural
    ):
        return [], word_chars

    # Don't include standalone umlaut chars in length
    num_umlauts = sum(1 for char in word_chars if char["text"] == "¨")
    end = math.ceil((len(word_chars) - num_umlauts) / 2)
    # Extend end by one for every standalone umlaut in emphasized part
    for _ in (char for char in word_chars[:end] if char["text"] == "¨"):
        while True:
            end += 1
            if word_chars[end - 1]["text"] != "¨":
                break

    return word_chars[:end], word_chars[end:]


def group_words(chars, offset_threshold=1.0, non_word_chars=""" \t,.!?;:/()[]{}<>§$%&_'"„“‚‘«»→—=""") -> list[list[dict]]:
    """
    Groups characters into words based on their positions and font properties.

    :param chars: List of pdfplumber character dictionaries
    :param offset_threshold: Distance threshold to consider characters as part of the same word (in points)
    :param non_word_chars: Word-breaking characters
    :return: List of words, each word as a list of character dictionaries
    """
    prev_char = None
    words = []
    word = []
    for char in chars:
        if word and prev_char:
            is_newline = prev_char["y0"] > char["y1"]
            was_hyphen = prev_char["text"] == "-"

            if (
                is_newline != was_hyphen or  # Unhyphenated new line or hyphen but no new line
                char["x0"] - prev_char["x1"] > offset_threshold or  # Space between words
                (not was_hyphen and abs(char["x0"] - prev_char["x1"]) > offset_threshold and
                 abs(char["matrix"][5] - prev_char["matrix"][5]) > offset_threshold) or  # New paragraph
                char["text"] in non_word_chars  # Word-breaking char
                # Cannot check for change of font, e.g., there are documents which use a different font for ligatures
            ):
                words.append(word)
                word = []

        if char["text"] != "-" and char["text"] not in non_word_chars:
            word.append(char)
        prev_char = char
    if word:
        words.append(word)
    return words


def _disassemble_ligatures(chars, overlay_font_name):
    """Disassembles ligatures into individual, correctly positioned characters."""
    result_chars = []
    for char in chars:
        # Ligatures mostly appear as char sequences, but sometimes as a single char
        char["text"] = char["text"].replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff").replace("ﬃ", "ffi")

        if len(char["text"]) > 1:
            strides = fonts.get_ligature_strides(char["text"], overlay_font_name)
            # Disassemble
            for s, c in zip(strides, char["text"]):
                new_char = char.copy()
                new_char["text"] = c
                new_char["x0"] += s * char["size"]
                result_chars.append(new_char)
        else:
            result_chars.append(char)

    return result_chars


def _draw_bbox(canvas: Canvas, char_lines: list, remaining_chars: list, top_pad_ratio=0.02):
    """
    Draws a filled bounding box on the characters to hide them (per line).
    :param top_pad_ratio: Additionally moves the top upwards to compensate for the under-estimated height of some fonts.
    """
    # Draw one box per line
    for i, line_chars in enumerate(char_lines):
        left = min(char["x0"] for char in line_chars)
        right = max(char["x1"] for char in line_chars)
        top = max(char["y1"] for char in line_chars)
        bottom = min(char["y0"] for char in line_chars)

        # Avoid drawing over the next character when their bounding boxes overlap
        if (
            i == len(char_lines) - 1 and remaining_chars and
            remaining_chars[0]["matrix"][5] == line_chars[-1]["matrix"][5] and
            remaining_chars[0]["x0"] < right
        ):
            right = remaining_chars[0]["x0"]

        if top_pad_ratio:
            top += (top - bottom) * top_pad_ratio

        canvas.setFillColorRGB(255, 255, 255)
        canvas.rect(left, bottom, right - left, top - bottom, stroke=0, fill=1)
        canvas.setFillColorRGB(0, 0, 0)


def _set_text_color(canvas: Canvas, color: tuple):
    """
    Sets the text color to whatever the PDF specifies.
    """
    if color is None:
        canvas.setFillColorRGB(0, 0, 0)
    elif len(color) == 1:
        canvas.setFillColorRGB(color[0], color[0], color[0])
    elif len(color) == 3:
        canvas.setFillColorRGB(*color)
    elif len(color) == 4:
        canvas.setFillColorCMYK(*color)
    else:
        canvas.setFillColorRGB(0, 0, 0)


def _get_char_lines(chars) -> list[list[dict]]:
    """Groups chars by y in case chars are on multiple lines."""
    y_chars = collections.defaultdict(list)
    for char in chars:
        y_chars[char["matrix"][5]].append(char)
    return list(y_chars.values())


def _iter_offset_chars(line_chars, overlay_font, dy_mode="median"):
    """
    Generates (x, y, char) tuples using the optimization offsets.
    :param dy_mode: How to position characters vertically. Either "median" (const y) or "individual" (per char).
    """
    for char in line_chars:
        x = char["x0"]
        y = char["matrix"][5]
        character = char["text"]
        if not overlay_font.get("char_offsets"):
            dx, dy = 0, 0
        else:
            offset_char = CHAR_MAP.get(character, character)
            dx, dy = overlay_font["char_offsets"].get(offset_char, (0, 0))
        if dy_mode == "median" and (median_y_offset := overlay_font.get("median_y_offset")) is not None:
            dy = median_y_offset
        yield (x + dx, y - dy, character)  # reportlab y axis is down-top


def _iter_rearranged_chars(line_chars, overlay_font):
    """
    Generates (x, y, char) tuples, rearranging characters horizontally based on their width.
    """
    widths = [fonts.get_char_width(char["text"], overlay_font["name"], overlay_font["size"]) for char in line_chars]
    if len(line_chars) > 2:
        available_width = line_chars[-1]["x1"] - line_chars[0]["x0"]
        chars_width = sum(widths)
        padding = (available_width - chars_width) / (len(line_chars) - 1)
    for i, char in enumerate(line_chars):
        if i == 0:
            x = char["x0"]
        elif i == len(line_chars) - 1:
            x = char["x1"] - widths[i]
        else:
            x += widths[i - 1] + padding
        y = char["matrix"][5]
        c = char["text"]
        yield (x, y, c)


def _draw_page_overlay(canvas: Canvas, page: pdfplumber.pdf.Page, remapped_fonts: dict = None, config=None):
    """
    Draws the overlay for the page to the canvas.

    :param canvas: Reportlab canvas object to draw on
    :param page: Pdfplumber page object
    :param remapped_fonts: Dictionary of font names to replace
    :param config: See `DEFAULT_CONFIG`
    """
    remapped_fonts = remapped_fonts or {}
    config = {key: (config[key] if config and key in config else value) for key, value in DEFAULT_CONFIG.items()}
    original_config = dict(config)

    font_names = set()
    missing_fonts = set()
    total_words = 0
    successful_words = 0

    # reportlab starts with its default font on every page
    current_font = None
    current_font_is_valid = False

    words = group_words(page.chars)
    for word in words:
        word_str = "".join(char["text"] for char in word)  # useful when debugging

        # if word_str == "quantification":
        #     print("break")

        chars, remaining_chars = _split_emphasized_part(word)
        if not chars:
            continue

        total_words += 1
        font_size = chars[0]["size"]
        font_name = chars[0]["fontname"]

        if (font_name, font_size) != current_font:
            current_font = (font_name, font_size)
            font_names.add(font_name)
            if font_name in remapped_fonts:
                font_name = remapped_fonts[font_name]
            overlay_font = fonts.setup_boldened_font(canvas, font_name, font_size, config["use_extrabold"])
            if overlay_font["state"] != "ok":
                if overlay_font["state"] == "missing":
                    missing_fonts.add(overlay_font["name"])
                current_font_is_valid = False
                continue
            current_font_is_valid = True
            config = dict(original_config)
            if overlay_font.get("config"):
                config.update(overlay_font["config"])

        if not current_font_is_valid:
            continue

        chars = _disassemble_ligatures(chars, overlay_font["name"])
        char_lines = _get_char_lines(chars)

        if config["draw_bbox"]:
            _draw_bbox(canvas, char_lines, remaining_chars)

        for line_chars in char_lines:
            if config["typesetting_mode"] == "full_offset":
                typeset_chars = _iter_offset_chars(line_chars, overlay_font, dy_mode="individual")
            elif config["typesetting_mode"] == "x_offset":
                typeset_chars = _iter_offset_chars(line_chars, overlay_font)
            elif config["typesetting_mode"] == "rearranged":
                typeset_chars = _iter_rearranged_chars(line_chars, overlay_font)
            else:
                raise ValueError(f"Unknown typesetting mode: {config['typesetting_mode']}")

            _set_text_color(canvas, line_chars[0]["non_stroking_color"])

            for x, y, c in typeset_chars:
                canvas.drawString(x, y, c)

        successful_words += 1

    return {
        "total_words": total_words,
        "successful_words": successful_words,
        "font_names": font_names,
        "missing_fonts": missing_fonts,
    }


def run_font_estimation(pdf: pdfplumber.PDF, pdf_font_name: str, samples=16):
    """
    Runs estimator.py in a subprocess, such that memory allocated by torch is freed as soon as we're done.
    Pytorch is currently only needed to estimate the primary font.
    """
    import pickle, time

    t0 = time.time()
    samples = list(extract.sample_crops(pdf, pdf_font_name, samples))
    t1 = time.time()
    logging.info(f"Memory usage after rendering samples: {get_memory_usage_mb()}")

    p = subprocess.Popen(["python", "ml/font/estimator.py", "--"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    pickle.dump(samples, p.stdin)
    p.stdin.close()
    result = p.stdout.readline()
    p.stdout.close()
    p.wait()
    logging.info(f"Sampling time: {t1 - t0:.2f}s, estimation time: {time.time() - t1:.2f}s")

    return {pdf_font_name: result.decode("utf-8").strip()}


def generate_text_overlay(input_pdf_file: IO | str):
    """
    Creates a temporary overlay pdf file and returns a metadata dict containing its path.
    """
    logging.info(f"Memory usage before reading document: {get_memory_usage_mb()}")
    logging.info(f"Top memory users: {get_top_memory_users_mb()}")

    font_names = set()
    missing_fonts = set()
    total_words = 0
    successful_words = 0

    logging.info("Reading document.")
    remapped_fonts = None
    with pdfplumber.open(input_pdf_file) as input_pdf:
        # Check if font names are encrypted
        primary_font = extract.get_primary_font(input_pdf)
        if primary_font is None or _is_encrypted_font(primary_font):
            logging.info(f"Memory usage before font estimation: {get_memory_usage_mb()}")
            logging.info(f"Top memory users: {get_top_memory_users_mb()}")
            remapped_fonts = run_font_estimation(input_pdf, primary_font)
            logging.info(f"Memory usage after font estimation: {get_memory_usage_mb()}")
            logging.info(f"Top memory users: {get_top_memory_users_mb()}")
        else:
            remapped_fonts = {}

        # Determine page size
        page_sizes = sorted((page.width, page.height) for page in input_pdf.pages)
        median_page_size = page_sizes[len(page_sizes) // 2]

        # Create overlay document
        with tempfile.NamedTemporaryFile(delete=False, suffix="_overlay.pdf") as overlay_pdf:
            canvas = Canvas(overlay_pdf, pagesize=median_page_size)

            # for page in tqdm(input_pdf.pages, "Generating overlay pages", file=sys.stderr):
            for page_number, page in enumerate(input_pdf.pages):
                page_result = _draw_page_overlay(canvas, page, remapped_fonts)
                canvas.showPage()  # new page
                font_names |= page_result["font_names"]
                missing_fonts |= page_result["missing_fonts"]
                total_words += page_result["total_words"]
                successful_words += page_result["successful_words"]
                logging.info(f"Memory usage after generating page {page_number}: {get_memory_usage_mb()}")
            canvas.save()

    logging.info(f"Document fonts: {font_names}")
    logging.info(f"Missing fonts: {missing_fonts}")

    success_ratio = successful_words / total_words if total_words > 0 else 0
    has_encrypted_fonts = any(_is_encrypted_font(f) for f in missing_fonts)
    font_estimation = {"estimated_primary_font": list(remapped_fonts.values())[0]} if remapped_fonts else {}
    summary = "warning" if success_ratio < 0.5 else "ok"
    primary_font = fonts.disambiguate_identifier(primary_font) if primary_font else None

    return {
        "path": overlay_pdf.name,
        "summary": summary,
        "total_words": total_words,
        "successful_words": successful_words,
        "success_ratio": success_ratio,
        "primary_font": primary_font,
        "has_encrypted_fonts": has_encrypted_fonts,
        **font_estimation
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import sys, json
    args = sys.argv[1:]

    if not len(args) == 1:
        print(f"Usage: {sys.argv[0]} <input_pdf_file>")
        sys.exit(1)

    metadata = generate_text_overlay(args[0])
    json.dump(metadata, sys.stdout, indent=2)
