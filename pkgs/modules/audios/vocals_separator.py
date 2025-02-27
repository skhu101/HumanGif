import os
from pathlib import Path

from audio_separator.separator import Separator


class VocalsSeparator:
    def __init__(self, output_dir: str, model_file: str) -> None:
        self.model_file = model_file
        self.output_dir = output_dir
        self.separator = Separator(output_dir=output_dir, output_single_stem="vocals", model_file_dir=self.model_file_dir)

    def load_model(self):
        self.separator.load_model(self.model_name)

    def separate(self, audio_file: str, output_file: str):
        output_file_path = Path(output_file)
        output_dir = output_file_path.parent
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.separator.output_dir = output_dir

        output = self.separator.separate(audio_file)

        os.rename(os.path.join(output_dir, output[0]), output_file)

        return output_file

    @property
    def model_file_dir(self):
        return Path(self.model_file).parent

    @property
    def model_name(self):
        return Path(self.model_file).name
