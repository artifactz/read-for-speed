from typing import TextIO, BinaryIO
import logging, sys
import PIL.Image
import pypdfium2


def _process(path: str, input: TextIO, output: BinaryIO, resolution: float):  # NOTE: not used
    """
    Reads page numbers from input, writes pickled raw bytes to output.
    """
    import pickle

    r = PdfRenderer(path, resolution)
    while (line := input.readline()):
        page_number = int(line.strip())
        image = r.render_page(page_number)
        image = image.convert("RGB")
        pickle.dump(image.size, output)
        pickle.dump(image.mode, output)
        pickle.dump(image.tobytes(), output)
        output.flush()


class PdfRenderer:
    def __init__(self, path: str, resolution: float):
        self.doc = pypdfium2.PdfDocument(path)
        self.resolution = resolution

    def render_page(self, page_number: int) -> PIL.Image.Image:  # NOTE: not used
        page = self.doc.get_page(page_number)
        bitmap = page.render(scale=self.resolution / 72, prefer_bgrx=True)
        return bitmap.to_pil()

    def render_page_rect(self, page_number: int, rect: tuple) -> PIL.Image.Image:
        page = self.doc.get_page(page_number)
        rect = (rect[0], rect[1], page.get_width() - rect[2], page.get_height() - rect[3])
        bitmap: pypdfium2.PdfBitmap = page.render(scale=self.resolution / 72, crop=rect, prefer_bgrx=True)

        # to_pil() calls PIL's frombuffer() which uses the buffer directly, so we shouldn't close bitmap here
        return bitmap.to_pil()


if __name__ == "__main__":
    # Reads page numbers from stdin, outputs pickled png bytes on stdout
    # NOTE: not used

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import sys
    args = sys.argv[1:]

    if len(args) != 2:
        print(f"Usage: {sys.argv[0]} <input_pdf_file> <resolution>", file=sys.stderr)
        sys.exit(1)

    _process(args[0], sys.stdin, sys.stdout.buffer, float(args[1]))
