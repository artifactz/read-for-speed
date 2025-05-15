import sys, os, collections, logging, multiprocessing, datetime, json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ml.font import extract
import util.tile_image


EVALUATION_DATA_PATH = "ml/font/evaluation_data"
EVALUATION_PDFS_PATH = "ml/font/evaluation_pdfs.json"
MODEL_TYPE = "ONNX"  # or "TORCH"


def run_crop_data_eval(model, batch_size: int = 1024):
    """Runs an evaluation of the model on the evaluation_data folder."""
    crops, labels = load_crop_data()
    output_chunks = []
    for i in tqdm(range(0, len(crops), batch_size), desc="Evaluating batches"):
        start = i
        end = min(i + batch_size, len(crops))
        batch = crops[start:end]
        output_chunks.append(model.predict(batch))
    outputs = np.concatenate(output_chunks, axis=0)
    hits, misses = verify_outputs(outputs, model.classes, labels)

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

    return zip(*samples)


def verify_outputs(outputs, classes, labels):
    """Compares the model outputs with the labels and counts hits and misses."""
    hits, misses = collections.Counter(), collections.Counter()

    for sample_outputs, label in zip(outputs, labels):
        predicted = sample_outputs.argmax()
        predicted = classes[predicted]
        if predicted == label:
            hits[label] += 1
        else:
            misses[label] += 1

    return hits, misses


def calculate_stats(hits, misses):
    classes = set(hits.keys()) | set(misses.keys())
    class_accs = {c: (hits[c] / (hits[c] + misses[c])) if hits[c] + misses[c] > 0 else None for c in classes}

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
    pool = multiprocessing.Pool(14, initializer=_initialize_pool_process)
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

    prediction = _model.predict_from_samples(samples)
    label = fonts.disambiguate_identifier(primary_font_raw)

    if prediction != label:
        folder = Path(out_folder) / path.stem
        folder.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            sample.save(folder / f"{i:02d}.png")
        with open(folder / "misclassification.json", "w") as f:
            json.dump({"path": str(path), "label": label, "prediction": prediction}, f, indent=4)

    return label, prediction


_model = None

def _initialize_pool_process():
    global _model
    if _model is None:
        _model = _load_model()
    # Also suppress pdfminer warnings
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)
    sys.stderr = open(os.devnull, 'w')


def _load_model(model_path: str = "ml/font/model.onnx", classes_path: str = "ml/font/classes.json"):
    if MODEL_TYPE == "TORCH":
        import estimator_torch
        return estimator_torch.Model(model_path, classes_path)
    elif MODEL_TYPE == "ONNX":
        import estimator_onnx
        return estimator_onnx.Model(model_path, classes_path)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")


if __name__ == "__main__":
    # run_crop_data_eval(model=_load_model("ml/font/old_models/2025-05-14/model.onnx", "ml/font/old_models/2025-05-14/classes.json"))
    run_crop_data_eval(model=_load_model())
    # run_pdf_files_eval(num_samples=4)
