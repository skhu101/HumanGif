
import json
import sys
from pathlib import Path
from typing import Text, Union

import librosa
import torch
from constants import CACHE_DIR
from pyannote.audio import Pipeline


class OverlapDetector:
    def __init__(
        self,
        checkpoint_path: Union[Text, Path],
        hparams_file: Union[Text, Path] = None,
        use_auth_token: Union[Text, None] = None,
        cache_dir: Union[Path, Text] = CACHE_DIR,
    ) -> None:
        self.pipeline = Pipeline.from_pretrained(checkpoint_path, hparams_file, use_auth_token, cache_dir)

    def load_model(self, device="cuda:0"):
        self.pipeline.to(torch.device(device))
        return self

    def process(self, audio_file: str, output_json_file: str=None):
        diarization = self.pipeline(audio_file)

        overlap_ranges: list[tuple[float, float]] = []
        for speech in diarization.get_timeline().support():
            overlap_ranges.append((float(speech.start), float(speech.end)))

        duration = librosa.get_duration(filename=audio_file)
        no_overlap_ranges: list[tuple[float, float]] = [(0, duration)]
        for overlap_range in overlap_ranges:
            end = no_overlap_ranges[-1][1]
            no_overlap_ranges[-1] = (no_overlap_ranges[-1][0], overlap_range[0])
            no_overlap_ranges.append((overlap_range[1], end))
        no_overlap_ranges = list(filter(lambda rang: rang[1] - rang[0] > 0, no_overlap_ranges))

        res = {
            "audio_file": audio_file,
            "no_overlap_ranges": no_overlap_ranges,
        }
        if output_json_file:
            with open(output_json_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(res))

        return res
