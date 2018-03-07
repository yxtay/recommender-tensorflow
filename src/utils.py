from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).parent.parent.resolve()


def make_dirs(path: str, file=False, empty: bool=False) -> Path:
    """
    create dir and clear if required
    """
    path = Path(path)
    dir_path = path.parent if file else path
    dir_path.mkdir(parents=True, exist_ok=True)

    if empty:
        for item in dir_path.glob("*"):
            if item.isfile():
                item.unlink()
    return path


def path_join(*paths: Iterable[str], file=False, empty: bool=False) -> Path:
    """
    join paths and create dir
    """
    path = Path(*paths)
    make_dirs(path, file, empty)
    return path
