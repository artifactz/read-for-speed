import json, multiprocessing, logging, os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import pdfplumber
import fonts
from ml.font import extract


QA_FILE = "ml/font/qa.json"


def run(fonts=None):
    """
    Runs QA for pdfs in the database that haven't been reviewed yet.

    :param fonts: List of font names to include in the QA. If None, all fonts are included.
    """
    from ml.font import pdf_data

    pdfs = pdf_data.get_pdf_files()

    queue = multiprocessing.Queue(10)
    pool = multiprocessing.Pool(8, initializer=_silence_pdfminer)
    num_queued = 0
    for i, entry in enumerate(pdfs):
        if (fonts and entry["primary_font"] not in fonts) or (entry["qa_status"] and entry["qa_status"] != "skipped"):
            continue

        pool.apply_async(_generate_crops, (entry["path"],), callback=queue.put)
        num_queued += 1

    num_done = len(pdfs) - num_queued
    pbar = tqdm(total=len(pdfs), initial=num_done)
    while True:
        path_str, primary_font_raw, primary_font, tabularized = queue.get()

        if fonts and primary_font not in fonts:
            status = "skipped"
            # print(f"Skipping {path_str} ({primary_font})")

        else:
            while True:
                cv2.imshow(f"{primary_font}  [O] ok  [S] scan  [W] weird  [B] browser  [Q] quit", tabularized)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == ord("o"):
                    status = "ok"
                    break
                elif key == ord("s"):
                    status = "scan"
                    break
                elif key == ord("w"):
                    status = "weird"
                    break
                elif key == ord("b"):
                    os.startfile(Path(path_str).resolve())
                    continue
                elif key == ord("q"):
                    queue.close()
                    print("Exiting...")
                    pool.terminate()
                    print("Waiting for processes to finish...")
                    pool.join()
                    print("Joined")
                    return

        pdf_data.update_pdf_file(path_str, qa_status=status)
        pbar.update(1)


def load_qa():
    try:
        with open(QA_FILE) as f:
            data = json.load(f)
        return data
    except OSError:
        return {}


def save_qa(data):
    with open(QA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _generate_crops(path):  # TODO pass primary_font_raw?
    with pdfplumber.open(path) as pdf:
        primary_font_raw = extract.get_primary_font(pdf)
        primary_font = fonts.disambiguate_identifier(primary_font_raw)
        crops = extract.sample_crops(pdf, primary_font_raw, 16)
        tabularized = extract.tabularize_crops([np.array(c) for c in crops])
    return path, primary_font_raw, primary_font, tabularized


def _silence_pdfminer():
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)


if __name__ == "__main__":
    run(fonts=["TimesNewRoman", "Arial", "Helvetica"])
    print("Done")

    # from ml.font import pdf_data
    # qa = load_qa()
    # for path, values in qa.items():
    #     pdf_data.update_pdf_file(path, qa_status=values["status"])
