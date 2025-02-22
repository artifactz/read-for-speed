import math, re, os
import pdfplumber
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import fonts


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


def get_emphasized_part(word_chars):
    if len(word_chars) < 2:
        return []

    word = "".join(char["text"] for char in word_chars)
    if re.match(r"^\d+(?:\w|\w\w)?$", word) or word == word.upper():
        return []

    # TODO: don't count standalone umlaut chars into length, but still emphasize them
    return word_chars[:math.ceil(len(word_chars) / 2)]


def group_words(chars, offset_threshold=1.0, non_word_chars=""" ,.!?;:/()[]{}<>§$%&_'"„“‚‘«»→—="""):
    prev_char = None
    words = []
    word = []
    for char in chars:
        if word and prev_char:
            is_newline = prev_char["y0"] > char["y1"]
            was_hyphen = prev_char["text"] == "-"
            
            if (
                is_newline != was_hyphen or  # Unhyphenated new line or hyphen but no new line
                char["x0"] - prev_char["x1"] > offset_threshold or  # Gap between words
                char["text"] in non_word_chars  # Word-breaking char
            ):
                words.append(word)
                word = []

        if char["text"] != "-" and char["text"] not in non_word_chars:
            word.append(char)
        prev_char = char
    if word:
        words.append(word)
    return words


def add_text_overlay(input_pdf_path, output_pdf_path, use_extrabold=False):
    """Adds text overlay to the input PDF and saves as output PDF."""

    # Temporary file for overlay
    overlay_pdf = "overlay.pdf"
    font_names = set()

    with pdfplumber.open(input_pdf_path) as pdf:
        # Create an overlay PDF
        c = canvas.Canvas(overlay_pdf, pagesize=letter)

        for page_number, page in enumerate(pdf.pages):
            words = group_words(page.chars)
            for word in words:
                word_str = "".join(char["text"] for char in word)

                # if word_str == "Amt":
                #     print("break")

                chars = get_emphasized_part(word)
                if not chars:
                    continue

                font_size = chars[0]["size"]
                font_name = chars[0]["fontname"]
                font_names.add(font_name)

                # if font_name == 'CVDRPC+DGMetaSerifScience-Italic':
                #     print("break")

                if not fonts.setup_boldened_font(c, font_name, font_size, use_extrabold):
                    continue
                for char in chars:
                    x = char["x0"]
                    y = char["matrix"][5]
                    character = char["text"]
                    if c._char_offsets is None:
                        dx, dy = 0, 0
                    else:
                        offset_char = CHAR_MAP.get(character, character)
                        dx, dy = c._char_offsets.get(offset_char, (0, 0))
                    c.drawString(x + dx, y + dy, character)

            c.showPage()  # new page

        c.save()

    print("Document fonts:", font_names)
    print("Missing fonts:", fonts._missing_fonts)

    # Merge overlay with the original pages
    writer = PdfWriter()
    reader = PdfReader(input_pdf_path)
    with open(overlay_pdf, "rb") as overlay_file:
        overlay_reader = PdfReader(overlay_file)
        for page_number, page in enumerate(reader.pages):
            overlay_page = overlay_reader.pages[page_number]
            page.merge_page(overlay_page)
            writer.add_page(page)

    # Save the output PDF
    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)

    # Clean up temporary overlay file
    os.remove(overlay_pdf)


if __name__ == "__main__":
    input_pdf_path = "sample10.pdf"
    output_pdf_path = "output10.pdf"
    add_text_overlay(input_pdf_path, output_pdf_path)

    print(f"Overlay added successfully. Saved as {output_pdf_path}")
