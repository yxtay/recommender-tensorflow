from pathlib import Path
import shutil

PROJECT_DIR = Path(__file__).parent.parent.resolve()


def make_dirs(path: str, isfile=False, empty: bool=False) -> Path:
    """
    create dir and clear if required
    """
    path = Path(path)
    dir_path = path.parent if isfile else path
    dir_path.mkdir(parents=True, exist_ok=True)

    if empty:
        shutil.rmtree(dir_path, ignore_errors=True)
    return path
