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
from src.utils import PROJECT_DIR, make_dirs

logger = get_logger(__name__)

FILES_CONFIG = {
    "users": {"filename": "u.user", "sep": "|", "columns": ["user_id", "age", "gender", "occupation", "zipcode"]},
    "items": {"filename": "u.item", "sep": "|",
              "columns": ["item_id", "title", "release", "video_release", "imdb", "unknown", "action", "adventure",
                          "animation", "children", "comedy", "crime", "documentary", "drama", "fantasy", "filmnoir",
                          "horror", "musical", "mystery", "romance", "scifi", "thriller", "war", "western"]},
    "all": {"filename": "u.data", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
    "train": {"filename": "ua.base", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
    "test": {"filename": "ua.test", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
}

DATA_DEFAULTS = {
    "data_dir": "data/ml-100k",
    "label": "label",
    "user_features": ["user_id", "age", "gender", "occupation", "zipcode", "zipcode1", "zipcode2", "zipcode3"],
    "item_features": ["item_id", "action", "adventure", "animation", "children", "comedy", "crime",
                      "documentary", "drama", "fantasy", "filmnoir", "horror", "musical", "mystery",
                      "romance", "scifi", "thriller", "war", "western", "release_year"],
    "context_features": ["year", "month", "day", "week", "dayofweek"],
    "dtype": {"zipcode": object, "zipcode1": object, "zipcode2": object, "zipcode3": object}
}
DATA_DEFAULTS.update({
    "all_csv": str(Path(DATA_DEFAULTS["data_dir"], "all.csv")),
    "train_csv": str(Path(DATA_DEFAULTS["data_dir"], "train.csv")),
    "test_csv": str(Path(DATA_DEFAULTS["data_dir"], "test.csv")),
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
            for item, conf in FILES_CONFIG.items()}

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
        context["label"] = context["rating"] - 1
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
    make_dirs(save_dir)

    for name, df in dfs.items():
        # save csv
        save_path = str(Path(save_dir, name + ".csv"))
        df.compute().to_csv(save_path, index=False)
        logger.info("data saved: %s.", save_path)
        # save tfrecord
        dd_tfrecord(df, str(Path(save_dir, name + ".tfrecord")))


def build_categorical_columns(df: dd.DataFrame,
                              user_features: Iterable[str] = DATA_DEFAULTS["user_features"],
                              item_features: Iterable[str] = DATA_DEFAULTS["item_features"],
                              context_features: Iterable[str] = DATA_DEFAULTS["context_features"]) -> Dict:
    columns_dict = {
        col: dd_create_categorical_column(df, col, num_oov_buckets=1)
        for col in ["user_id", "age", "gender", "occupation", "zipcode", "zipcode1", "zipcode2", "zipcode3",
                    "item_id", "unknown", "action", "adventure", "animation", "children", "comedy", "crime",
                    "documentary", "drama", "fantasy", "filmnoir", "horror", "musical", "mystery", "romance",
                    "scifi", "thriller", "war", "western", "release_year",
                    "year", "month", "day", "week", "dayofweek"]
        }
    columns_dict["age_bucket"] = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("age", dtype=tf.int32),
        [15, 25, 35, 45, 55, 65]
    )
    columns_dict["release_year_bucket"] = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("release_year", dtype=tf.int32),
        [1930, 1940, 1950, 1960, 1970, 1980, 1990]
    )

    user_columns = [columns_dict[col] for col in user_features]
    item_columns = [columns_dict[col] for col in item_features]
    context_columns = [columns_dict[col] for col in context_features]

    return {"user_columns": user_columns, "item_columns": item_columns, "context_columns": context_columns}


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
        data_dir = str(Path(args.dest, "ml-100k"))
        download_data(args.url, args.dest)
        data = load_data(data_dir)
        dfs = process_data(data)
        save_data(dfs, data_dir)

    except Exception as e:
        logger.exception(e)
        raise e
