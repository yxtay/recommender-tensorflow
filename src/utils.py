import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()


def make_dirs(path, isfile=False, empty=False):
    """
    create dir and clear if required
    """
    path = Path(path)
    dir_path = path.parent if isfile else path
    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    if empty:
        shutil.rmtree(dir_path, ignore_errors=True)
    return path
