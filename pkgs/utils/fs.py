
import os
from typing import Generator


def traverse_folder(folder: str, deep=True, filter_fn=lambda file: True) -> Generator[str, None, None]:
    assert os.path.exists(folder)
    assert os.path.isdir(folder)

    sub_items = map(
        lambda item: os.path.join(folder, item),
        os.listdir(folder),
    )

    sub_items = sorted(sub_items, key=lambda item: os.stat(item).st_size)

    for item_path in sub_items:
        if os.path.isdir(item_path):
            if deep:
                yield from traverse_folder(item_path, deep, filter_fn)
        else:
            if filter_fn(item_path):
                yield os.path.abspath(item_path)
