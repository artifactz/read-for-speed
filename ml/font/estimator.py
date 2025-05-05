import json, sys
from typing import Iterable
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from ml.font import extract, train


with open("ml/font/classes.json") as f:
    classes = json.load(f)

model = train.Net(classes)
model.load_state_dict(torch.load("ml/font/model.pth"))
if torch.cuda.is_available():
    model.to(torch.device('cuda:0'))
model.eval()


def estimate_primary_font(pdf: "pdfplumber.PDF", samples=24) -> tuple[str, str]:
    """
    Estimates the primary font used in a PDF document.
    :param pdf: pdfplumber pdf object to analyze
    :param samples: number of samples to take from the document
    :return: tuple of (predicted font name, font name used in the document)
    """
    sampler = extract.CropSampler(pdf)
    samples = sampler.sample_iter(samples)
    prediction = estimate_primary_font_from_samples(samples)
    return prediction, sampler.primary_font_raw


def estimate_primary_font_from_samples(samples: Iterable) -> str:
    """
    Estimates the primary font used in a list of samples.
    :param samples: list of PIL Image samples to analyze
    :return: predicted font name
    """
    tensors = [train.img_to_tensor(img) for img in samples]
    tensors = torch.stack(tensors)
    with torch.no_grad():
        outputs = model(tensors)
        summed = outputs.sum(dim=0).cpu().numpy()
        predicted = summed.argmax()
    return classes[predicted]


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        pdf_file = "samples/encrypted/sample31.pdf"
    elif args[0] == "--":
        import pickle
        samples = pickle.load(sys.stdin.buffer)
        prediction = estimate_primary_font_from_samples(samples)
        print(prediction)
        exit(0)
    else:
        pdf_file = " ".join(args)

    import pdfplumber, json
    with pdfplumber.open(pdf_file) as pdf:
        predicted, primary_font_name = estimate_primary_font(pdf)
        print(json.dumps({primary_font_name: predicted}))
