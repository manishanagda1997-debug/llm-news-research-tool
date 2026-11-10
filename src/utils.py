"""Small helpers shared by several files."""

from pathlib import Path


def unique_path(folder: Path, stem: str, suffix: str = ".json") -> Path:
    """
    A file path in `folder` that is not taken yet.

        tata-motors.json      (if free)
        tata-motors_2.json    (if the first is taken)
        tata-motors_3.json    ...

    The time in the name alone is not enough: on Windows the clock can
    give the same time to two saves made one after another.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}{suffix}"
    number = 2
    while path.exists():
        path = folder / f"{stem}_{number}{suffix}"
        number += 1
    return path
