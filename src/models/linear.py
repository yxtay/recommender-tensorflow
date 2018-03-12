from argparse import ArgumentParser
import shutil
import sys

import dask.dataframe as dd
import tensorflow as tf

from src.data.ml_100k import DATA_DEFAULTS, build_categorical_columns
from src.logger import get_logger
from src.tf_utils import tf_csv_dataset
from src.utils import PROJECT_DIR


def train_main(args):
    # build feature columns
    df = dd.read_csv(args.train_csv, dtype=DATA_DEFAULTS["dtype"]).persist()
    categorical_columns = build_categorical_columns(df, feature_names=DATA_DEFAULTS["feature_names"])
    linear_columns = [tf.feature_column.indicator_column(col)
                      for col in categorical_columns]

    # clean up model directory
    shutil.rmtree(args.model_dir, ignore_errors=True)
    # build model
    model = tf.estimator.LinearClassifier(
        feature_columns=linear_columns,
        model_dir=args.model_dir,
    )

    logger.debug("model training started.")
    for n in range(args.num_epochs):
        # train model
        model.train(input_fn=lambda: tf_csv_dataset(args.train_csv, DATA_DEFAULTS["label"], shuffle=True))
        # evaluate model
        results = model.evaluate(input_fn=lambda: tf_csv_dataset(args.test_csv, DATA_DEFAULTS["label"]))
        logger.info("epoch %s: %s.", n, results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train-csv", default=str(PROJECT_DIR / DATA_DEFAULTS["train_csv"]),
                        help="path to the training csv data (default: %(default)s)")
    parser.add_argument("--test-csv", default=str(PROJECT_DIR / DATA_DEFAULTS["test_csv"]),
                        help="path to the test csv data (default: %(default)s)")
    parser.add_argument("--model-dir", default="checkpoints/linear",
                        help="model directory (default: %(default)s)")
    parser.add_argument("--num-epochs", type=int, default=16,
                        help="number of training epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size (default: %(default)s)")
    parser.add_argument("--log-path", default=str(PROJECT_DIR / "main.log"),
                        help="path of log file (default: %(default)s)")
    args = parser.parse_args()

    logger = get_logger(__name__, log_path=args.log_path, console=True)
    logger.debug("call: %s.", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s.", args)
    tf.logging.set_verbosity(tf.logging.INFO)

    try:
        train_main(args)
    except Exception as e:
        logger.exception(e)
        raise e
