import json
import torch
import pdfplumber
from ml.font import extract, train


with open("ml/font/classes.json") as f:
    classes = json.load(f)
    classes = {v: k for k, v in classes.items()}

model = train.Net(len(classes))
model.load_state_dict(torch.load("ml/font/model.pth"))
model.eval()


def estimate_primary_font(pdf: pdfplumber.PDF, samples=32) -> tuple[str, str]:
    """
    Estimates the primary font used in a PDF document.
    :param pdf: pdfplumber pdf object to analyze
    :param samples: number of samples to take from the document
    :return: tuple of (predicted font name, font name used in the document)
    """
    sampler = extract.CropSampler(pdf)
    samples = [sampler.sample() for _ in range(samples)]
    tensors = [train.img_to_tensor(img) for img in samples]
    tensors = torch.stack(tensors)
    with torch.no_grad():
        outputs = model(tensors)
        summed = outputs.sum(dim=0).cpu().numpy()
        predicted = summed.argmax()
    predicted = classes[predicted]
    return predicted, sampler.primary_font_raw


if __name__ == "__main__":
    pdf_path = "samples/encrypted/sample31.pdf"
    with pdfplumber.open(pdf_path) as pdf:
        print(estimate_primary_font(pdf))
