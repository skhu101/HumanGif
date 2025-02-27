
from typing import Iterable


def get_data_frames(data: Iterable, parallelism: int, rank: int):
    pick = 0
    data_frames = []
    for item in data:
        if pick % parallelism == rank:
            data_frames.append(item)
            print(f"PICK: {item}")
            print(f"PICKED_LEN: {len(data_frames)}")
        pick += 1

    return data_frames
