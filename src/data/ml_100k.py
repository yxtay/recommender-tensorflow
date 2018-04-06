from argparse import ArgumentParser
import sys
from pathlib import Path
import shutil
from typing import Dict, Iterable
from zipfile import ZipFile

import dask.dataframe as dd
import requests
import tensorflow as tf

from src.logger import get_logger
from src.tf_utils import dd_tfrecord, dd_create_categorical_column

logger = get_logger(__name__)

FILE_CONFIG = {
    "users": {"filename": "u.user", "sep": "|", "columns": ["user_id", "age", "gender", "occupation", "zipcode"]},
    "items": {"filename": "u.item", "sep": "|",
              "columns": ["item_id", "title", "release", "video_release", "imdb", "unknown", "action", "adventure",
                          "animation", "children", "comedy", "crime", "documentary", "drama", "fantasy", "filmnoir",
                          "horror", "musical", "mystery", "romance", "scifi", "thriller", "war", "western"]},
    "all": {"filename": "u.data", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
    "train": {"filename": "ua.base", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
    "test": {"filename": "ua.test", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
}

DATA_DEFAULT = {
    "data_dir": "data/ml-100k",
    "label": "label",
    "categorical_columns": ["user_id", "age", "gender", "occupation", "zipcode", "zipcode1", "zipcode2", "zipcode3",
                            "item_id", "unknown", "action", "adventure", "animation", "children", "comedy", "crime",
                            "documentary", "drama", "fantasy", "filmnoir", "horror", "musical", "mystery", "romance",
                            "scifi", "thriller", "war", "western", "release_year",
                            "year", "month", "day", "week", "dayofweek"],
    "feature_names": ["user_id", "age", "gender", "occupation", "zipcode", "zipcode1", "zipcode2", "zipcode3",
                      "item_id", "action", "adventure", "animation", "children", "comedy", "crime",
                      "documentary", "drama", "fantasy", "filmnoir", "horror", "musical", "mystery",
                      "romance", "scifi", "thriller", "war", "western", "release_year",
                      "year", "month", "day", "week", "dayofweek"],
    "dtype": {"zipcode": object, "zipcode1": object, "zipcode2": object, "zipcode3": object}
}
DATA_DEFAULT.update({
    "all_csv": str(Path(DATA_DEFAULT["data_dir"], "all.csv")),
    "train_csv": str(Path(DATA_DEFAULT["data_dir"], "train.csv")),
    "test_csv": str(Path(DATA_DEFAULT["data_dir"], "test.csv"))
})


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
            for item, conf in FILE_CONFIG.items()}

    logger.info("data loaded.")
    return data


def process_data(data: Dict[str, dd.DataFrame]) -> Dict[str, dd.DataFrame]:
    # process users
    users = data["users"]
    users["zipcode1"] = users["zipcode"].str.get(0)
    users["zipcode2"] = users["zipcode"].str.slice(0, 2)
    users["zipcode3"] = users["zipcode"].str.slice(0, 3)
    data["users"] = users.persist()
    logger.debug("users data processed.")

    # process items
    items = data["items"]
    items = items[items["title"] != "unknown"]  # remove "unknown" movie
    items["release"] = dd.to_datetime(items["release"])
    items["release_year"] = items["release"].dt.year
    data["items"] = items.persist()
    logger.debug("items data processed.")

    # process context
    for el in ["all", "train", "test"]:
        context = data[el]
        context["label"] = (context["rating"] >= 4).astype(int)
        context["datetime"] = dd.to_datetime(context["timestamp"], unit="s")
        context["year"] = context["datetime"].dt.year
        context["month"] = context["datetime"].dt.month
        context["day"] = context["datetime"].dt.day
        context["week"] = context["datetime"].dt.week
        context["dayofweek"] = context["datetime"].dt.dayofweek + 1
        data[el] = context
    logger.debug("context data processed.")

    # merge data
    dfs = {item: (data[item]
                  .merge(data["users"], "inner", "user_id")
                  .merge(data["items"], "inner", "item_id")
                  .persist())
           for item in ["all", "train", "test"]}
    logger.info("data merged.")

    return dfs


def save_data(dfs: Dict[str, dd.DataFrame], save_dir: str = "data/ml-100k") -> None:
    for name, df in dfs.items():
        # save csv
        save_path = str(Path(save_dir, name + ".csv"))
        df.compute().to_csv(save_path, index=False)
        logger.info("data saved: %s.", save_path)
        # save tfrecord
        dd_tfrecord(df, str(Path(save_dir, name + ".tfrecord")))


def build_categorical_columns(df: dd.DataFrame,
                              feature_names: Iterable[str] = DATA_DEFAULT["feature_names"]) -> Iterable:
    # categorical columns
    columns_dict = {
        col: dd_create_categorical_column(df, col)
        for col in DATA_DEFAULT["categorical_columns"]
    }

    # bucketized columns
    columns_dict["age_bucket"] = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("age", dtype=tf.int32),
        [15, 25, 35, 45, 55, 65]
    )
    columns_dict["release_year_bucket"] = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("release_year", dtype=tf.int32),
        [1930, 1940, 1950, 1960, 1970, 1980, 1990]
    )

    categorical_columns = [columns_dict[col] for col in feature_names]
    return categorical_columns


if __name__ == "__main__":
    parser = ArgumentParser(description="Download, extract and prepare MovieLens 100k data.")
    parser.add_argument("--url", default="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                        help="url of MovieLens 100k data (default: %(default)s)")
    parser.add_argument("--dest", default="data",
                        help="destination directory for downloaded and extracted files (default: %(default)s)")
    parser.add_argument("--log-path", default="main.log",
                        help="path of log file (default: %(default)s)")
    args = parser.parse_args()

    logger = get_logger(__name__, log_path=args.log_path, console=True)
    logger.debug("call: %s.", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s.", args)

    try:
        data_dir = str(Path(args.dest, "ml-100k"))
        download_data(args.url, args.dest)
        data = load_data(data_dir)
        dfs = process_data(data)
        save_data(dfs, data_dir)

    except Exception as e:
        logger.exception(e)
        raise e
