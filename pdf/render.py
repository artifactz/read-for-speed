import PIL.Image
import pypdfium2


class PdfRenderer:
    def __init__(self, path: str, resolution: float):
        self.doc = pypdfium2.PdfDocument(path)
        self.resolution = resolution

    def render_page_rect(self, page_number: int, rect: tuple) -> PIL.Image.Image:
        page = self.doc.get_page(page_number)
        rect = (rect[0], rect[1], page.get_width() - rect[2], page.get_height() - rect[3])
        bitmap: pypdfium2.PdfBitmap = page.render(scale=self.resolution / 72, crop=rect, prefer_bgrx=True)

        # to_pil() calls PIL's frombuffer() which uses the buffer directly, so we shouldn't close bitmap here
        return bitmap.to_pil()
