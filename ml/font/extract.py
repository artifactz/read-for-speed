import sys, collections, logging, multiprocessing.pool, random, shutil, json
from pathlib import Path
import PIL.Image
import numpy as np
from tqdm import tqdm
import pdfplumber, pdfplumber.page

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import fonts, util.tile_image


def index_primary_fonts(folder, update_existing=False):
    """Stores the primary font of each PDF file in the database."""
    from ml.font import pdf_data
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)

    paths = [str(path).replace("\\", "/") for path in Path(folder).glob("*.pdf")]
    existing_entries = [pdf_data.get_pdf_file(path) for path in paths]

    if not update_existing:
        paths = [path for path, entry in zip(paths, existing_entries) if not entry]
        existing_entries = [None] * len(paths)

    pool = multiprocessing.pool.Pool(16, initializer=_silence_pdfminer)
    for path, existing_entry, primary_font_raw in zip(
        paths, existing_entries, tqdm(pool.imap(get_primary_font_from_path, paths), total=len(paths))
    ):
        primary_font = fonts.disambiguate_identifier(primary_font_raw) if primary_font_raw else None
        if existing_entry:
            pdf_data.update_pdf_file(path, primary_font=primary_font, primary_font_raw=primary_font_raw)
        else:
            pdf_data.add_pdf_file(path, primary_font=primary_font, primary_font_raw=primary_font_raw)

    pool.close()
    pool.join()


def generate_training_data_from_database(
    output_folder: str,
    total_samples: int,
    classes=["ComputerModernSerif", "TimesNewRoman", "NimbusRomNo9L", "P052", "MinionPro", "STIXTwoText", "WarnockPro",
             "Arial", "Calibri", "AGaramond", "LinLibertine", "Cambria", "PTSerif", "Helvetica"]
):
    """
    Generates training data by sampling random crops from PDF documents and saving them as images. Balances the number
    of samples per document to ensure a similar number of samples for each font class.

    :param primary_fonts_path: Path or list of paths to primary_fonts.txt files, which contain paths to PDF documents
                               and their primary font names
    :param output_folder: Folder to save the generated images (creates font_name/document_name subfolders)
    :param total_samples: Total number of samples to generate
    :param enable_skip: If True, skips documents that have already been processed
    :param classes: List of font classes to include in the training data
    """
    from ml.font import pdf_data
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)
    path_fonts = []
    num_documents = collections.Counter()

    for pdf_entry in pdf_data.get_pdf_files():
        path = pdf_entry["path"]
        font = pdf_entry["primary_font"]
        if not font in classes or (pdf_entry["qa_status"] in ["scan", "weird"]):
            continue
        path_fonts.append((path, font))
        num_documents[font] += 1

    num_samples_per_class = total_samples / len(classes)
    class_num_samples = {k: round(num_samples_per_class / v) for k, v in num_documents.items()}
    args = [(path, output_folder, class_num_samples[font], False) for path, font in path_fonts]
    random.shuffle(args)

    pool = multiprocessing.pool.Pool(16, initializer=_silence_pdfminer)
    for _ in tqdm(pool.imap_unordered(_generate_training_data_star, args), total=len(path_fonts)):
        pass
    pool.close()
    pool.join()
    # for arg in args:  # for debugging: without multiprocessing
    #     generate_training_data(*arg)


def _generate_training_data_star(args):
    generate_training_data(*args)


def generate_training_data(pdf_path: str, output_folder: str | Path, num_samples: int = 100, verbose=True):
    """
    Generates training data by sampling random crops from PDF documents and saving them as PNG images.

    :param pdf_path: Path to PDF document
    :param output_folder: Folder to save the generated images (creates font_name/document_name subfolders)
    :param num_samples: Number of crops to generate
    """
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            primary_font_raw = get_primary_font(pdf)
            primary_font = fonts.disambiguate_identifier(primary_font_raw)
            folder = output_folder / primary_font / pdf_path.stem
            folder.mkdir(parents=True, exist_ok=True)

            sample_gen = sample_crops(pdf, primary_font_raw, num_samples)
            if verbose:
                sample_gen = tqdm(sample_gen, desc=f"Processing {pdf_path.stem}", total=num_samples, unit="sample")
            with util.tile_image.tile_image_writer(folder) as writer:
                for img in sample_gen:
                    writer.write(img)

    except OSError as e:
        print(f"Error reading file {pdf_path.name}: {e}")


def sample_crops(pdf: pdfplumber.PDF, primary_font_raw: str, n: int, rect_size: float = 64, k: int = 30,
                 crop_size: int = 128, dpi: float = 600):
    """
    Samples random crops from a PDF document.

    :param pdf: pdfplumber PDF object
    :param primary_font_raw: pdf font name of preferred font
    :param n: number of crops to sample
    :param rect_size: size of the rectangle to sample (in PDF points)
    :param k: number of attempts to sample a rectangle
    :param crop_size: size of the resulting crops (in pixels)
    """
    from pdf import render

    page_rects = sorted(sample_page_rect(pdf, primary_font_raw, rect_size, k) for _ in range(n))
    renderer = render.PdfRenderer(pdf.path, dpi)

    for page_number, rect in page_rects:
        img = renderer.render_page_rect(page_number, rect)
        img = img.convert("L")
        img = img.resize((crop_size, crop_size))
        img = normalize_crop(img)
        yield img


def normalize_crop(image: PIL.Image.Image, invert_threshold=127) -> PIL.Image.Image:
    """Stretches colorspace to 0..255 and inverts the image if it's too dark."""
    changed = False
    arr = np.array(image, dtype=float)
    mini, maxi = arr.min(), arr.max()
    color_range = maxi - mini
    if not (color_range == 0 or (mini == 0 and maxi == 255)):
        arr = (arr - mini) / color_range * 255
        changed = True
    if np.mean(arr) < invert_threshold:
        arr = 255 - arr
        changed = True
    if not changed:
        return image
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(arr, mode="L")


def get_primary_font_from_path(path: str | Path) -> str:
    """Returns the most used font in a PDF document specified by path."""
    try:
        with pdfplumber.open(path) as pdf:
            return get_primary_font(pdf)
    except Exception as e:
        print(f"Error reading file {path}: {e}")


def get_primary_font(pdf: pdfplumber.PDF) -> str:
    """Returns the most used font in a PDF document object."""
    font_counts = count_font_characters(pdf)
    if not font_counts:
        return None
    return font_counts.most_common(1)[0][0]


def count_font_characters(pdf: pdfplumber.PDF) -> collections.Counter:
    """Counts the number of characters in each font on each page of a PDF document."""
    result = collections.Counter()
    for page in pdf.pages:
        for char in page.chars:
            font = char["fontname"]
            result[font] += 1
    return result


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


def split_train_eval(training_folder: str | Path, evaluation_folder: str | Path, eval_pdfs_file: str | Path):
    """
    Splits the training data into training and evaluation sets by moving some of the document folders in the training
    folder to the evaluation folder.

    :param training_folder: Folder containing the training data
    :param evaluation_folder: Folder to save the evaluation data
    :param eval_pdfs_file: File containing the list of PDF files to use for evaluation
    """
    with open(eval_pdfs_file) as f:
        eval_pdfs = json.load(f)
    training_folder = Path(training_folder)
    evaluation_folder = Path(evaluation_folder)
    for font_folder in training_folder.iterdir():
        doc_folders = list(font_folder.iterdir())
        for path in doc_folders:
            if str(path).replace("\\", "/") in eval_pdfs:
                new_path = evaluation_folder / path.relative_to(training_folder)
                shutil.move(path, new_path)


def show_crops(pdf_path, n: int):
    """Generates random crops from a PDF document and displays them in a grid."""
    import cv2
    with pdfplumber.open(pdf_path) as pdf:
        primary_font_raw = get_primary_font(pdf)
        crops = list(sample_crops(pdf, primary_font_raw, n))
        tabularized = tabularize_crops([np.array(c) for c in crops])
        cv2.imshow(primary_font_raw, tabularized)
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


def _silence_pdfminer():
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)


if __name__ == "__main__":
    # show_crops("samples/arxiv/2003.08388v1.pdf", 25)
    # index_primary_fonts("samples/arxiv", update_existing=False)
    # index_primary_fonts("samples/scholar", update_existing=False)

    generate_training_data_from_database("ml/font/training_data", 200000)
    split_train_eval("ml/font/training_data", "ml/font/evaluation_data", "ml/font/evaluation_pdfs.json")

    # with pdfplumber.open("samples/output75.pdf") as pdf:
    #     print(get_primary_font(pdf))
