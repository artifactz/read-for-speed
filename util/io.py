from typing import Iterable
from pathlib import Path


def load_file_values(paths: str | Path | Iterable[str | Path]) -> list[tuple[str, str]]:
    """
    Loads key-value pairs from a txt file or multiple txt files, where each key refers to a file in the same directory
    as the txt file. The txt file only stores the file names, which are then expanded to full paths.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    pairs = []
    for path in paths:
        path = Path(path)
        with open(path, "r") as f:
            for line in f.readlines():
                k, v = [s.strip() for s in line.split(":")]
                k = str(path.parent / k)
                pairs.append((k, v))

    return pairs
