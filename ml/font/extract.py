import sys, collections, uuid, logging
from pathlib import Path
import PIL.Image
import numpy as np
from tqdm import tqdm
import pdfplumber, pdfplumber.page

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import fonts


def generate_training_data(pdf_paths: list, output_folder: str, num_samples_per_document: int = 100):
    """
    Generates training data by sampling random crops from PDF documents and saving them as images.

    :param pdf_paths: list of paths to PDF documents
    :param output_folder: folder to save the generated images (creates font_name/document_name subfolders)
    """
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        with pdfplumber.open(pdf_path) as pdf:
            sampler = CropSampler(pdf)
            folder = Path(output_folder) / Path(sampler.primary_font) / pdf_path.stem
            folder.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(range(num_samples_per_document), desc=f"Processing {pdf_path.stem}", unit="sample"):
                img = sampler.sample()
                filename = f"{uuid.uuid4()}.png"
                img.save(folder / filename)


class CropSampler:
    """Samples random crops from a PDF document as PIL images. Caches rendered pages."""
    def __init__(self, pdf: pdfplumber.PDF, crop_size: int = 128, rect_size: float = 64, k: int = 30, dpi: float = 600):
        """
        Initializes the CropSampler with a PDF document and parameters for sampling.

        :param pdf: pdfplumber PDF object
        :param crop_size: size of the resulting crops (in pixels)
        :param rect_size: size of the rectangle to sample (in PDF points)
        :param k: number of attempts to sample a rectangle
        :param dpi: resolution for rendering the PDF pages
        """
        self.pdf = pdf
        self.crop_size = crop_size
        self.rect_size = rect_size
        self.dpi = dpi
        self.k = k
        self.page_images = [None] * len(pdf.pages)
        self.primary_font_raw = count_font_characters(pdf).most_common(1)[0][0]
        self.primary_font = fonts._disambiguate_identifier(self.primary_font_raw)

    def sample(self) -> PIL.Image.Image:
        """Samples a random crop from the PDF document."""
        page_number, rect = sample_page_rect(self.pdf, self.primary_font_raw, self.rect_size, self.k)
        page = self.pdf.pages[page_number]
        if self.page_images[page_number] is None:
            self.page_images[page_number] = page.to_image(self.dpi).original
        img: PIL.Image.Image = self.page_images[page_number]

        # Convert to image coordinates
        img_rect = (
            rect[0] / page.width * img.width,
            (1 - (rect[3] / page.height)) * img.height,
            rect[2] / page.width * img.width,
            (1 - (rect[1] / page.height)) * img.height,
        )
        # Crop the image
        img_cropped = img.crop(img_rect)
        # Convert to grayscale
        img_cropped = img_cropped.convert("L")
        # Resize to crop size
        img_cropped = img_cropped.resize((self.crop_size, self.crop_size))

        return img_cropped


def count_font_characters(pdf: pdfplumber.PDF) -> collections.Counter:
    """Counts the number of characters in each font on each page of a PDF document."""
    result = collections.Counter()
    for page in pdf.pages:
        for char in page.chars:
            font = char["fontname"]
            result[font] += 1
    return result


def get_chars_in_rect(page: pdfplumber.page.Page, rect: tuple) -> list[dict]:
    """
    Returns all characters that overlap with a given rectangle on a page.

    :param page: pdfplumber page object
    :param rect: tuple of (x0, y0, x1, y1) coordinates defining the rectangle
    """
    x0, y0, x1, y1 = rect
    chars_in_rect = [
        char for char in page.chars
        if char["x1"] > x0 and char["x0"] < x1 and char["y1"] > y0 and char["y0"] < y1
    ]
    return chars_in_rect


def sample_page_rect(pdf: pdfplumber.PDF, preferred_font: str, size: float, k: int) -> tuple[int, tuple]:
    """
    Randomly samples a rectangle from a pdf document. Tries k times and returns the sample that maximizes:

    * the number of characters of the preferred font
    * the number of unique characters

    and minimizes the number of characters of other fonts.

    :param pdf: pdfplumber PDF object
    :param preferred_font: font name to maximize
    :param size: size of the rectangle to sample
    :param k: number of attempts to sample a rectangle
    :return: tuple of (page number, rectangle coordinates)
    """
    page_numbers = np.random.randint(0, len(pdf.pages), size=k)
    rects = []
    scores = []
    for p in page_numbers:
        x0 = np.random.randint(0, pdf.pages[p].width - size)
        y0 = np.random.randint(0, pdf.pages[p].height - size)
        rect = (x0, y0, x0 + size, y0 + size)
        rects.append(rect)
        chars = get_chars_in_rect(pdf.pages[p], rect)
        # Score
        ignored_chars = "."  # avoid table of contents
        font_counts = collections.Counter(char["fontname"] for char in chars if char["text"] not in ignored_chars)
        other_font_counts = sum(font_counts.values()) - font_counts[preferred_font]
        unique_chars = len(set(char["text"] for char in chars))
        score = font_counts[preferred_font] - other_font_counts + unique_chars
        scores.append(score)
    best_index = np.argmax(scores)
    return page_numbers[best_index], rects[best_index]


def show_crops(pdf_path, n: int, *args, **kwargs):
    """Generates random crops from a PDF document and displays them in a grid."""
    import cv2
    with pdfplumber.open(pdf_path) as pdf:
        sampler = CropSampler(pdf, *args, **kwargs)
        crops = [sampler.sample() for _ in range(n)]
        tabularized = tabularize_crops([np.array(c) for c in crops])
        cv2.imshow(sampler.primary_font, tabularized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def tabularize_crops(crops: list[np.ndarray]) -> np.ndarray:
    """Tabularizes a list of crops into a grid."""
    n = len(crops)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    tabularized = np.zeros((rows * crops[0].shape[0], cols * crops[0].shape[1]), dtype=crops[0].dtype)
    for i, crop in enumerate(crops):
        row = i // cols
        col = i % cols
        tabularized[row * crop.shape[0]:(row + 1) * crop.shape[0], col * crop.shape[1]:(col + 1) * crop.shape[1]] = crop
    return tabularized


def print_primary_fonts(folder):
    """Prints the primary font of each PDF file in the given folder."""
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)
    for path in Path(folder).glob("sample*.pdf"):
        with pdfplumber.open(path) as pdf:
            font_counts = count_font_characters(pdf)
            if not font_counts:
                print(f"{path.name}: No fonts found")
                continue
            primary_font = fonts._disambiguate_identifier(font_counts.most_common(1)[0][0])
            print(f"{path.name}: {primary_font}")


if __name__ == "__main__":
    # print_primary_fonts("samples")
    show_crops("samples/sample8.pdf", 12)

    # generate_training_data(
    #     [
    #         # Times New Roman
    #         "samples/sample16.pdf",
    #         "samples/sample26.pdf",
    #         "samples/sample27.pdf",
    #         "samples/sample36.pdf",
    #         "samples/sample4.pdf",
    #         "samples/sample41.pdf",
    #         "samples/sample47.pdf",
    #         "samples/sample48.pdf",
    #         "samples/sample57.pdf",
    #         "samples/sample62.pdf",
    #         "samples/sample7.pdf",

    #         # ComputerModernSerif
    #         "samples/sample12.pdf",
    #         "samples/sample18.pdf",
    #         "samples/sample5.pdf",
    #         "samples/sample50.pdf",
    #         "samples/sample55.pdf",
    #         "samples/sample56.pdf",
    #         "samples/sample58.pdf",
    #         "samples/sample59.pdf",
    #         "samples/sample60.pdf",

    #         # Arial
    #         "samples/sample3.pdf",
    #         "samples/synthetic/Arial01_Volkshochschulen.pdf",
    #         "samples/synthetic/Arial02_Terraria.pdf",

    #         # AGaramond
    #         "samples/sample1.pdf",
    #         "samples/sample46.pdf",

    #         # Verdana
    #         "samples/sample10.pdf",
    #         "samples/sample9.pdf",

    #         # Helvetica
    #         "samples/sample2.pdf",
    #         "samples/sample45.pdf",

    #         # LinLibertine
    #         "samples/sample20.pdf",
    #         "samples/sample21.pdf",
    #     ],
    #     "ml/font/training_data"
    # )
