import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile

import dask.dataframe as dd
import requests

from src.gcp_utils import (get_bigquery_client, df_to_bigquery,
                           bigquery_to_table, bigquery_to_gcs)
from src.logger import get_logger
from src.utils import make_dirs

logger = get_logger(__name__)

DATA_CONFIG = {
    "users": {"filename": "u.user", "sep": "|", "columns": ["user_id", "age", "gender", "occupation", "zipcode"]},
    "items": {"filename": "u.item", "sep": "|",
              "columns": ["item_id", "title", "release", "video_release", "imdb", "unknown", "action", "adventure",
                          "animation", "children", "comedy", "crime", "documentary", "drama", "fantasy", "filmnoir",
                          "horror", "musical", "mystery", "romance", "scifi", "thriller", "war", "western"]},
    "all": {"filename": "u.data", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
    "train": {"filename": "ua.base", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
    "test": {"filename": "ua.test", "sep": "\t", "columns": ["user_id", "item_id", "rating", "timestamp"]},
}


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
            for item, conf in DATA_CONFIG.items()}

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
    items["release_date"] = dd.to_datetime(items["release"])
    items["release_year"] = items["release_date"].dt.year
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


def bigquery_process_data(dataset, client):
    # process users
    users_query = (
        "SELECT "
        "   user_id, age, gender, occupation, zipcode, "
        "   SUBSTR(zipcode, 0, 1) AS zipcode1,"
        "   SUBSTR(zipcode, 0, 2) AS zipcode2,"
        "   SUBSTR(zipcode, 0, 3) AS zipcode3 "
        "FROM {dataset}.users"
    ).format(dataset=dataset)
    bigquery_to_table(users_query, "users_full", dataset, client)
    logger.info("users processed.")

    # process items
    items_query = (
        "SELECT "
        "   item_id, title, release, video_release, imdb, "
        "   unknown, action, adventure, animation, children, comedy, "
        "   crime, documentary, drama, fantasy, filmnoir, horror, "
        "   musical, mystery, romance, scifi, thriller, war, western, "
        "   PARSE_DATE('%d-%b-%Y', release) AS release_date, "
        "   EXTRACT(YEAR FROM PARSE_DATE('%d-%b-%Y', release)) AS release_year "
        "FROM {dataset}.items "
        "WHERE title != 'unknown'"
    ).format(dataset=dataset)
    bigquery_to_table(items_query, "items_full", dataset, client)
    logger.info("items processed.")

    # process context and join users, items
    for table in ["all", "train", "test"]:
        context_query = (
            "SELECT "
            "   user_id, item_id, rating, timestamp, "
            "   TIMESTAMP_SECONDS(timestamp) AS datetime, "
            "   EXTRACT(YEAR FROM TIMESTAMP_SECONDS(timestamp)) as year, "
            "   EXTRACT(MONTH FROM TIMESTAMP_SECONDS(timestamp)) as month, "
            "   EXTRACT(DAY FROM TIMESTAMP_SECONDS(timestamp)) as day, "
            "   EXTRACT(ISOWEEK FROM TIMESTAMP_SECONDS(timestamp)) as week, "
            "   EXTRACT(DAYOFWEEK FROM TIMESTAMP_SECONDS(timestamp)) as dayofweek, "
            "   age, gender, occupation, zipcode, zipcode1, zipcode2, zipcode3, "
            "   title, release, video_release, imdb, "
            "   unknown, action, adventure, animation, children, comedy, "
            "   crime, documentary, drama, fantasy, filmnoir, horror, "
            "   musical, mystery, romance, scifi, thriller, war, western, "
            "   release_date, release_year "
            "FROM {dataset}.{table} "
            "JOIN {dataset}.users_features USING (user_id) "
            "JOIN {dataset}.items_features USING (item_id)"
        ).format(dataset=dataset, table=table)
        bigquery_to_table(context_query, table + "_full", dataset, client)
        logger.info("%s processed.", table)


def save_data(dfs, save_dir="data/ml-100k"):
    for name, df in dfs.items():
        # save csv
        save_path = str(Path(save_dir, name + ".csv"))
        df.compute().to_csv(save_path, index=False, encoding="utf-8")
        logger.info("data saved: %s.", save_path)


def local_main(args):
    url = args.url
    dest = args.dest

    download_data(url, dest)
    data_dir = str(Path(dest, "ml-100k"))
    data = load_data(data_dir)
    dfs = process_data(data)
    save_data(dfs, data_dir)


def gcp_main(args):
    url = args.url
    dest = args.dest
    dataset = args.dataset
    credentials = args.credentials
    bucket = args.gcs_bucket

    # download, unzip and load data
    download_data(url, dest)
    data_dir = str(Path(dest, "ml-100k"))
    data = load_data(data_dir)

    client = get_bigquery_client(credentials)

    # upload data to bigquery
    for name, df in data.items():
        df_to_bigquery(df, name, args.dataset, client)

    # process data with bigquery
    bigquery_process_data(args.dataset, client)

    # export bigquery tables to gcs
    for name in data:
        path = "{dest}/ml-100k/{table}.csv".format(dest=dest, table=name)
        bigquery_to_gcs(name + "_full", dataset, path, bucket, client)


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
    gcp_parser.add_argument("--gcs-bucket", default="recommender-tensorflow",
                            help="google cloud storage bucket to store processed files")
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
