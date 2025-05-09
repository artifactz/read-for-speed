import sys, os, collections, logging, multiprocessing, datetime, json
from pathlib import Path
import torch
from tqdm import tqdm
from ml.font import estimator, train, extract
import util.tile_image


EVALUATION_DATA_PATH = "ml/font/evaluation_data"
EVALUATION_PDFS_PATH = "ml/font/evaluation_pdfs.json"


def run_crop_data_eval(batch_size: int = 1024):
    """Runs an evaluation of the model on the evaluation_data folder."""
    tensors, labels = load_crop_data()

    if estimator.device:
        tensors = tensors.to(estimator.device)
    output_chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tensors), batch_size), desc="Evaluating batches"):
            start = i
            end = min(i + batch_size, len(tensors))
            batch = tensors[start:end]
            output_chunks.append(estimator.model(batch))
    outputs = torch.cat(output_chunks, dim=0).cpu().numpy()

    hits, misses = verify_outputs(outputs, labels)

    stats = calculate_stats(hits, misses)
    for c, acc in stats["class_accs"].items():
        print(f"{c} accuracy: {acc * 100:.2f}%")
    print(f"Total accuracy: {stats['total_accuracy'] * 100:.2f}%")
    print(f"Mean accuracy: {stats['mean_acc'] * 100:.2f}%")


def load_crop_data():
    """Reads images and labels from the evaluation_data folder."""
    samples = []
    for path, folders, files in Path(EVALUATION_DATA_PATH).walk():
        for file in files:
            if not file.endswith(".png"):
                continue
            for img in util.tile_image.read_tile_image(path / file, 128):
                label = path.parent.stem
                samples.append((img, label))

    tensors = [train.img_to_tensor(img) for img, _ in samples]
    tensors = torch.stack(tensors)
    labels = [label for _, label in samples]

    return tensors, labels


def verify_outputs(outputs, labels):
    """Compares the model outputs with the labels and counts hits and misses."""
    hits, misses = collections.Counter(), collections.Counter()

    for sample_outputs, label in zip(outputs, labels):
        predicted = sample_outputs.argmax()
        predicted = estimator.classes[predicted]
        if predicted == label:
            hits[label] += 1
        else:
            misses[label] += 1

    return hits, misses


def calculate_stats(hits, misses):
    class_accs = {c: (hits[c] / (hits[c] + misses[c])) if hits[c] + misses[c] > 0 else None for c in estimator.classes}

    total_hits = sum(hits.values())
    total_misses = sum(misses.values())
    total_accuracy = total_hits / (total_hits + total_misses)

    mean_acc = sum(class_accs.values()) / len(class_accs)

    return {
        "class_accs": class_accs,
        "total_accuracy": total_accuracy,
        "mean_acc": mean_acc,
        "correct": total_hits,
        "incorrect": total_misses,
        "total": total_hits + total_misses,
    }


def run_pdf_files_eval(num_samples: int):
    """Runs an evaluation of the model on the pdf files defined in `EVALUATION_PDFS_PATH`."""
    hits, misses = verify_pdf_files(EVALUATION_PDFS_PATH, num_samples)
    total = hits + misses
    print(f"Total accuracy ({num_samples} samples): {hits / total * 100:.2f}%  (correct: {hits}, incorrect: {misses})")


def verify_pdf_files(pdfs_file: str | Path, num_samples: int):
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_folder = f"out/evaluation/{timestamp}"

    with open(pdfs_file) as f:
        paths = json.load(f)

    hits, misses = 0, 0
    args = [(path, num_samples, out_folder) for path in paths]
    pool = multiprocessing.Pool(14, initializer=_silence_warnings)
    results = list(tqdm(pool.imap(_get_pdf_label_prediction_star, args), total=len(args), desc="Evaluating PDF files"))
    pool.close()
    pool.join()

    for label, prediction in results:
        if prediction == label:
            hits += 1
        else:
            misses += 1

    return hits, misses


def _get_pdf_label_prediction_star(args):
    return _get_pdf_label_prediction(*args)


def _get_pdf_label_prediction(path, num_samples, out_folder):
    import pdfplumber, fonts
    path = Path(path)

    with pdfplumber.open(path) as pdf:
        primary_font_raw = extract.get_primary_font(pdf)
        samples = list(extract.sample_crops(pdf, primary_font_raw, num_samples))

    prediction = estimator.estimate_primary_font_from_samples(samples)
    label = fonts.disambiguate_identifier(primary_font_raw)

    if prediction != label:
        folder = Path(out_folder) / path.stem
        folder.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            sample.save(folder / f"{i:02d}.png")
        with open(folder / "misclassification.json", "w") as f:
            json.dump({"path": str(path), "label": label, "prediction": prediction}, f, indent=4)

    return label, prediction


def _silence_warnings():
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)
    sys.stderr = open(os.devnull, 'w')


if __name__ == "__main__":
    # run_crop_data_eval()
    run_pdf_files_eval(num_samples=4)
