from argparse import ArgumentParser
import sys
from pathlib import Path
import shutil
from typing import Dict
from zipfile import ZipFile

import dask.dataframe as dd
import requests

from src.logger import get_logger
from src.utils import PROJECT_DIR, make_dirs

logger = get_logger(__name__)

FILES_CONFIG = {
    "events": {"filename": "u.data", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
    "users": {"filename": "u.user", "sep": "|", "columns": ["user_id", "age", "gender", "occupation", "zipcode"]},
    "items": {"filename": "u.item", "sep": "|",
              "columns": ["item_id", "title", "release", "video_release", "imdb", "unknown", "action", "adventure",
                          "animation", "children", "comedy", "crime", "documentary", "drama", "fantasy", "filmnoir",
                          "horror", "musical", "mystery", "romance", "scifi", "thriller", "war", "western"]},
}


def download_data(url: str = "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                  dest_dir: str = "data"):
    # prepare destination
    dest = Path(dest_dir) / Path(url).name
    dest.parent.mkdir(parents=True, exist_ok=True)

    # downlaod zip
    if not dest.exists():
        logger.info("downloading file: %s.", url)
        r = requests.get(url, stream=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(r.raw, f)
        logger.info("file downloaded: %s.", dest)

    # extract zip
    if not Path(dest_dir, "ml-100k", "README").exists():
        with dest.open("rb") as f, ZipFile(f, "r") as zf:
            zf.extractall(dest_dir)
        logger.info("file extracted.")


def load_data(src_dir: str = "data/ml-100k") -> Dict[str, dd.DataFrame]:
    data = {item: dd.read_csv(str(Path(src_dir, conf["filename"])), sep=conf["sep"],
                              header=None, names=conf["columns"], encoding="latin-1")
            for item, conf in FILES_CONFIG.items()}

    logger.info("data loaded.")
    return data


def process_data(data: Dict[str, dd.DataFrame], save_path: str = "data/ml-100k/processed.csv") -> dd.DataFrame:
    # process events
    events = data["events"]
    events["normalised_rating"] = events["rating"] / (events["rating"].max() - events["rating"].min())
    events["datetime"] = dd.to_datetime(events["timestamp"], unit="s")
    events["year"] = events["datetime"].dt.year
    events["month"] = events["datetime"].dt.month
    events["day"] = events["datetime"].dt.day
    events["week"] = events["datetime"].dt.week
    events["dayofweek"] = events["datetime"].dt.dayofweek
    data["events"] = events

    # process users
    users = data["users"]
    users["zipcode1"] = users["zipcode"].str.get(0)
    users["zipcode2"] = users["zipcode"].str.slice(0, 2)
    users["zipcode3"] = users["zipcode"].str.slice(0, 3)

    # process items
    items = data["items"]
    items["release"] = dd.to_datetime(items["release"])
    items["release_year"] = items["release"].dt.year

    # merge data
    df = (data["events"]
          .merge(data["users"], "left", "user_id")
          .merge(data["items"], "left", "item_id"))
    logger.info("data merged.")

    # save data
    if save_path:
        df.compute().to_csv(make_dirs(save_path, file=True), index=False)
        logger.info("data saved: %s.", save_path)
    return df


if __name__ == "__main__":
    parser = ArgumentParser(description="Download, extract and prepare MovieLens 100k data.")
    parser.add_argument("--url", default="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                        help="url of MovieLens 100k data (default: %(default)s)")
    parser.add_argument("--dest", default="data",
                        help="destination directory of downloaded and extracted files (default: %(default)s)")
    parser.add_argument("--log-path", default=str(PROJECT_DIR / "main.log"),
                        help="path of log file (default: %(default)s)")
    args = parser.parse_args()

    logger = get_logger(__name__, log_path=args.log_path, console=True)
    logger.debug("call: %s.", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s.", args)

    try:
        download_data(args.url, args.dest)
        data = load_data(Path(args.dest, "ml-100k"))
        process_data(data, Path(args.dest, "ml-100k", "processed.csv"))

    except Exception as e:
        logger.exception(e)
        raise e
