import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile

import dask.dataframe as dd
import requests
import tensorflow as tf

from src.gcp_utils import get_credentials, df_upload_bigquery
from src.logger import get_logger
from src.tf_utils import dd_create_categorical_column
from src.utils import make_dirs

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


def download_data(url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                  dest_dir="data"):
    # prepare destination
    dest = Path(dest_dir) / Path(url).name
    make_dirs(dest, isfile=True)

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


def load_data(src_dir="data/ml-100k"):
    data = {item: dd.read_csv(str(Path(src_dir, conf["filename"])), sep=conf["sep"],
                              header=None, names=conf["columns"], encoding="latin-1")
            for item, conf in FILE_CONFIG.items()}

    logger.info("data loaded.")
    return data


def process_data(data):
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


def save_data(dfs, save_dir="data/ml-100k"):
    for name, df in dfs.items():
        # save csv
        save_path = str(Path(save_dir, name + ".csv"))
        df.compute().to_csv(save_path, index=False, encoding="utf-8")
        logger.info("data saved: %s.", save_path)


def local_main(args):
    download_data(args.url, args.dest)
    data_dir = str(Path(args.dest, "ml-100k"))
    data = load_data(data_dir)
    dfs = process_data(data)
    save_data(dfs, data_dir)


def gcp_main(args):
    download_data(args.url, args.dest)
    data_dir = str(Path(args.dest, "ml-100k"))
    data = load_data(data_dir)
    credentials = get_credentials(args.credentials)

    for name, df in data.items():
        df_upload_bigquery(df, args.dataset, name, credentials)


if __name__ == "__main__":
    parser = ArgumentParser(description="Download, extract and prepare MovieLens 100k data.")
    subparsers = parser.add_subparsers(title="subcommands")

    # local download and preprocess
    local_parser = subparsers.add_parser("local")
    local_parser.add_argument("--url", default="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                              help="url of MovieLens 100k data (default: %(default)s)")
    local_parser.add_argument("--dest", default="data",
                              help="destination directory for downloaded and extracted files (default: %(default)s)")
    local_parser.add_argument("--log-path", default="main.log",
                              help="path of log file (default: %(default)s)")
    local_parser.set_defaults(main=local_main)

    # gcp upload
    gcp_parser = subparsers.add_parser("gcp")
    gcp_parser.add_argument("--url", default="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                            help="url of MovieLens 100k data (default: %(default)s)")
    gcp_parser.add_argument("--dest", default="data",
                            help="destination directory for downloaded and extracted files (default: %(default)s)")
    gcp_parser.add_argument("--dataset", default="ml_100k",
                            help="dataset name to save datatables")
    gcp_parser.add_argument("--credentials", default="credentials.json",
                            help="json file containing google cloud credentials")
    gcp_parser.add_argument("--log-path", default="main.log",
                            help="path of log file (default: %(default)s)")
    gcp_parser.set_defaults(main=gcp_main)

    args = parser.parse_args()

    logger = get_logger(__name__, log_path=args.log_path, console=True)
    logger.debug("call: %s.", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s.", args)

    try:
        args.main(args)
    except Exception as e:
        logger.exception(e)
        raise e
