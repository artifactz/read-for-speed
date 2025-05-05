import torch, collections
from pathlib import Path
from ml.font import estimator, train
import util.tile_image


EVALUATION_DATA_PATH = "ml/font/evaluation_data"


def run():
    tensors, labels = load_data()

    with torch.no_grad():
        outputs = estimator.model(tensors).cpu().numpy()

    hits, misses = verify_outputs(outputs, labels)

    stats = calculate_stats(hits, misses)
    for c, acc in stats["class_accs"].items():
        print(f"{c} accuracy: {acc * 100:.2f}%")
    print(f"Total accuracy: {stats['total_accuracy'] * 100:.2f}%")
    print(f"Mean accuracy: {stats['mean_acc'] * 100:.2f}%")


def load_data():
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


if __name__ == "__main__":
    run()
