import json
from pathlib import Path
from typing import Text, Union

import torch
from constants import CACHE_DIR
from pyannote.audio import Pipeline


class SingleVoiceSplitter:
    def __init__(
        self,
        checkpoint_path: Union[Text, Path],
        hparams_file: Union[Text, Path] = None,
        use_auth_token: Union[Text, None] = None,
        cache_dir: Union[Path, Text] = CACHE_DIR,
    ) -> None:
        self.pipeline = Pipeline.from_pretrained(checkpoint_path, hparams_file, use_auth_token, cache_dir)


    def load_model(
        self,
        device: str = "cuda:0"
    ):
        self.pipeline.to(torch.device(device))
        return self

    def process(self, audio_file: str, output_json_file: str=None, min_duration=0.5):

        diarization = self.pipeline(audio_file)

        segments: list[tuple[float, float]] = []
        prev = None
        start = 0.0
        turn = None
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if prev is None:
                prev = speaker
                start = turn.start
            elif prev != speaker:
                segments.append((start, turn.start))
                prev = speaker
                start = turn.start

        if turn:
            if turn.end:
                segments.append((start, turn.end))

        segments = list(filter(lambda x: x[1] - x[0] >= min_duration, segments))

        res = {
            "audio_file": audio_file,
            "segments": segments,
        }

        if output_json_file:
            with open(output_json_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(res))

        return res
