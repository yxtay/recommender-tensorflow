import shutil
from argparse import ArgumentParser

import tensorflow as tf

from trainers.conf_utils import get_run_config, get_train_spec, get_exporter, get_eval_spec
from trainers.ml_100k import get_feature_columns, get_input_fn, serving_input_fn


def train_and_evaluate(args):
    # paths
    train_csv = args.train_csv
    test_csv = args.test_csv
    job_dir = args.job_dir
    restore = args.restore
    # model
    embedding_size = args.embedding_size
    hidden_units = args.hidden_units
    dropout = args.dropout
    # training
    batch_size = args.batch_size
    train_steps = args.train_steps

    # init
    tf.logging.set_verbosity(tf.logging.INFO)
    if not restore:
        shutil.rmtree(job_dir, ignore_errors=True)

    # estimator
    feature_columns = get_feature_columns(embedding_size=embedding_size)
    run_config = get_run_config()
    estimator = tf.estimator.DNNClassifier(
        hidden_units=hidden_units,
        feature_columns=feature_columns["deep"],
        model_dir=job_dir,
        dropout=dropout,
        config=run_config
    )

    # train spec
    train_input_fn = get_input_fn(train_csv, batch_size=batch_size)
    train_spec = get_train_spec(train_input_fn, train_steps)

    # eval spec
    eval_input_fn = get_input_fn(test_csv, tf.estimator.ModeKeys.EVAL, batch_size=batch_size)
    exporter = get_exporter(serving_input_fn)
    eval_spec = get_eval_spec(eval_input_fn, exporter)

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train-csv", default="data/ml-100k/train.csv",
                        help="path to the training csv data (default: %(default)s)")
    parser.add_argument("--test-csv", default="data/ml-100k/test.csv",
                        help="path to the test csv data (default: %(default)s)")
    parser.add_argument("--job-dir", default="checkpoints/deep",
                        help="job directory (default: %(default)s)")
    parser.add_argument("--restore", action="store_true",
                        help="whether to restore from job_dir")
    parser.add_argument("--embedding-size", type=int, default=4,
                        help="embedding size (default: %(default)s)")
    parser.add_argument("--hidden-units", type=int, nargs='+', default=[16, 16],
                        help="hidden layer specification (default: %(default)s)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout rate (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size (default: %(default)s)")
    parser.add_argument("--train-steps", type=int, default=20000,
                        help="number of training steps (default: %(default)s)")
    args = parser.parse_args()

    train_and_evaluate(args)
