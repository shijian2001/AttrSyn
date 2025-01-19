import json
import os
from PIL import Image
import numpy as np

class Logger:
    def __init__(self, log_file="training_log.json", image_dir="generated_images"):
        self.log_file = log_file
        self.image_dir = image_dir
        self.ensure_log_file_exists()
        self.ensure_image_dir_exists()
        self.image_counter = 0

    def ensure_log_file_exists(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def ensure_image_dir_exists(self):
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def save_image(self, image):
        self.image_counter += 1
        image_name = f"image_{self.image_counter}.png"
        image_path = os.path.join(self.image_dir, image_name)
        image.save(image_path)
        return image_path

    def log(self, image, prompt, all_questions, all_answers, all_qa_results, reward_score):
        image_path = self.save_image(image)

        log_entry = {
            "image_path": image_path,
            "prompt": prompt,
            "questions": all_questions,
            "answers": all_answers,
            "qa_results": all_qa_results,
            "reward_score": self.convert_to_python_type(reward_score)
        }
        self.append_log(log_entry)

    def append_log(self, log_entry):
        with open(self.log_file, "r+") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=4)

    def convert_to_python_type(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, list):
            return [self.convert_to_python_type(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_python_type(value) for key, value in obj.items()}
        return obj
