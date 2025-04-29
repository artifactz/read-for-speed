import logging
from PyPDF2 import PdfReader, PdfWriter
from util.memory_usage import get_memory_usage_mb


def merge(input_pdf_file_1, input_pdf_file_2, output_pdf_file):
    """Merges two PDF files into one such that the second file is rendered on top of the first."""
    writer = PdfWriter()
    reader1 = PdfReader(input_pdf_file_1)
    reader2 = PdfReader(input_pdf_file_2)
    # for page_number, page in enumerate(tqdm(reader1.pages, "Merging overlay with original pages")):
    for page_number, page in enumerate(reader1.pages):
        overlay_page = reader2.pages[page_number]  # Fails with IndexError if input 2 has fewer pages
        page.merge_page(overlay_page)
        writer.add_page(page)
        logging.info(f"Memory usage after merging page {page_number}: {get_memory_usage_mb()}")

    _copy_metadata(reader1, writer)

    # Save the output PDF
    logging.info("Saving output document.")
    writer.write(output_pdf_file)


def _copy_metadata(reader: PdfReader, writer: PdfWriter):
    """
    Copies metadata. Skips non-string values to avoid PyPDF2 TypeError.
    (Reader.metadata may contain non-string values, such as list, but then writer crashes during `write`.)
    """
    meta = {}
    for k in reader.metadata:  # cannot iterate over items() because their values are IndirectObject instead of str
        if isinstance(value := reader.metadata[k], str):
            meta[k] = value
    writer.add_metadata(meta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import sys
    args = sys.argv[1:]

    if not len(args) == 3:
        print(f"Usage: {sys.argv[0]} <input_pdf_file_1> <input_pdf_file_2> <output_pdf_file>")
        sys.exit(1)

    merge(args[0], args[1], args[2])
    logging.info("Merged PDF files successfully.")
