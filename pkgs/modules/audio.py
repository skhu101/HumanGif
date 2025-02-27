import json
import os
import sys
from pathlib import Path

import torch
from audio_separator.separator import Separator
from constants import CACHE_DIR
from pyannote.audio import Pipeline


def parse_no_overlap(audio_file: str, output_json_file: str=None, device="cuda:0"):
    pipeline = Pipeline.from_pretrained(
        checkpoint_path="pretrained_models/overlap_detect/overlap_detect.yaml",
    )
    pipeline.to(torch.device(device))

    diarization = pipeline(audio_file)

    overlap_ranges: list[tuple[float, float]] = []
    for speech in diarization.get_timeline().support():
        overlap_ranges.append((float(speech.start), float(speech.end)))

    no_overlap_ranges: list[tuple[float, float]] = [(0, None)]
    for overlap_range in overlap_ranges:
        no_overlap_ranges[-1] = (no_overlap_ranges[-1][0], overlap_range[0])
        no_overlap_ranges.append((overlap_range[1], None))
    no_overlap_ranges[-1] = (no_overlap_ranges[-1][0], float(sys.maxsize))

    res = {
        "audio_file": audio_file,
        "no_overlap_ranges": no_overlap_ranges,
    }
    if output_json_file:
        with open(output_json_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(res))

    return res

def parse_single_voice(audio_file: str, output_json_file: str=None, min_duration=0.5, device="cuda:0"):
    pipeline = Pipeline.from_pretrained(
        checkpoint_path="pretrained_models/speaker-diarization/config.yaml",
    )
    pipeline.to(torch.device(device))
    diarization = pipeline(audio_file)

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

def separate_vocals(audio_file: str, output_file: str, model_file: str):
    model_file_path = Path(model_file)
    model_name = model_file_path.name
    model_dir = model_file_path.parent

    output_file_path = Path(output_file)
    output_dir = output_file_path.parent

    os.makedirs(output_dir, exist_ok=True)

    separator = Separator(model_file_dir=model_dir, output_single_stem="vocals", output_dir=output_dir)

    separator.load_model(model_name)

    output = separator.separate(audio_file)
    os.rename(os.path.join(output_dir, output[0]), output_file)

    return output_file

if __name__ == "__main__":
    # parse_single_voice("/home/leeway/workspace/data_processors/.cache/output/stage6/qgeUZDx7KHY_6_0-0-428.wav")
    separate_vocals("/home/leeway/workspace/data_processors/test_data/audios/t.wav", "/home/leeway/workspace/data_processors/test_data/audios/t_vocal.wav", "/home/leeway/workspace/data_processors/.cache/pretrained_models/uvr/Kim_Vocal_2.onnx")
