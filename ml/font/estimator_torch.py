from typing import Iterable
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms
from ml.font import train


class Model:
    def __init__(self, model_path="ml/font/model.pth", classes_path="ml/font/classes.json"):
        with open(classes_path) as f:
            self.classes = json.load(f)
        self.model = train.Net(self.classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else None
        if self.device:
            self.model.to(self.device)

    def predict(self, samples: Iterable[Image.Image]) -> np.ndarray:
        to_tensor = torchvision.transforms.ToTensor()
        tensors = [2 * (1 - to_tensor(img)) - 1 for img in samples]
        tensors = torch.stack(tensors)
        return self._predict(tensors)

    def _predict(self, input_data: torch.Tensor) -> np.ndarray:
        if self.device:
            input_data = input_data.to(self.device)
        with torch.no_grad():
            return self.model(input_data).cpu().numpy()

    def predict_from_samples(self, samples: Iterable[Image.Image]) -> str:
        outputs = self.predict(samples)
        summed = outputs.sum(axis=0)
        predicted = summed.argmax()
        return self.classes[predicted]
