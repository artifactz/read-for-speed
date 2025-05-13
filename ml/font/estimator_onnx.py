from typing import Iterable
import json
from PIL import Image
import numpy as np
import onnxruntime


class Model:
    def __init__(self, model_path="ml/font/model.onnx", classes_path="ml/font/classes.json"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        with open(classes_path) as f:
            self.classes = json.load(f)

    def predict(self, samples: Iterable[Image.Image]) -> np.ndarray:
        inputs = [2 * (1 - np.array(img, dtype=np.float32) / 255) - 1 for img in samples]
        inputs = np.stack(inputs)
        inputs = np.expand_dims(inputs, axis=1)
        return self._predict(inputs)

    def _predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: input_data})[0]

    def predict_from_samples(self, samples: Iterable[Image.Image]) -> str:
        """
        Estimates the primary font used in a list of samples.
        :param samples: list of PIL Image samples to analyze
        :return: predicted font name
        """
        outputs = self.predict(samples)
        summed = outputs.sum(axis=0)
        predicted = summed.argmax()
        return self.classes[predicted]


if __name__ == "__main__":
    import sys, pickle
    model = Model()
    samples = pickle.load(sys.stdin.buffer)
    prediction = model.predict_from_samples(samples)
    print(prediction)
