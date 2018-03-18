from argparse import ArgumentParser
import shutil
import sys
from typing import Dict

import dask.dataframe as dd
import tensorflow as tf

from src.data.ml_100k import DATA_DEFAULTS, build_categorical_columns
from src.logger import get_logger
from src.tf_utils import tf_csv_dataset, layer_summary, get_binary_predictions, get_binary_losses, get_binary_metrics
from src.utils import PROJECT_DIR


def model_fn(features: Dict[str, tf.Tensor], labels: tf.Tensor, mode, params: Dict) -> tf.estimator.EstimatorSpec:
    categorical_columns = params.get("categorical_columns", [])
    numeric_columns = params.get("numeric_columns", [])
    use_linear = params.get("use_linear", True)
    use_mf = params.get("use_mf", True)
    use_dnn = params.get("use_dnn", True)
    embedding_size = params.get("embedding_size", 16)
    hidden_units = params.get("hidden_units", [64, 64, 64])
    activation_fn = params.get("activation", tf.nn.relu)
    dropout = params.get("dropout", 0)
    learning_rate = params.get("learning_rate", 0.001)

    logits = 0
    if use_linear:
        with tf.variable_scope("linear"):
            linear_logit = tf.feature_column.linear_model(
                features,
                feature_columns=categorical_columns + numeric_columns,
                units=1
            )
            # [None, 1]

            with tf.name_scope("linear"):
                layer_summary(linear_logit)
            logits += linear_logit

    if use_mf or use_dnn:
        # create embedding input only if mf or dnn present
        embedding_columns = [tf.feature_column.embedding_column(col, embedding_size)
                             for col in categorical_columns]
        embedding_input = tf.feature_column.input_layer(features, embedding_columns)
        # [None, d * embedding_size]

        if use_mf:
            with tf.variable_scope("mf"):
                # reshape flat embedding input layer to matrix
                embedding_mat = tf.reshape(embedding_input, [-1, len(embedding_columns), embedding_size])
                # [None, d, embedding_size]
                sum_square = tf.square(tf.reduce_sum(embedding_mat, 1))
                # [None, embedding_size]
                square_sum = tf.reduce_sum(tf.square(embedding_mat), 1)
                # [None, embedding_size]

                with tf.name_scope("logits"):
                    mf_logit = tf.multiply(tf.reduce_sum(tf.subtract(sum_square, square_sum), 1, keepdims=True), 0.5)
                    # [None, 1]

                    layer_summary(mf_logit)
                logits += mf_logit

        if use_dnn:
            with tf.variable_scope("dnn/dnn"):
                net = embedding_input
                # [None, d * embedding_size]

                for i, hidden_size in enumerate(hidden_units):
                    with tf.variable_scope("hiddenlayer_%s" % i):
                        net = tf.layers.dense(net, hidden_size, activation=activation_fn)
                        # [None, hidden_size]
                        if dropout > 0 and mode == tf.estimator.ModeKeys.TRAIN:
                            net = tf.layers.dropout(net, rate=dropout, training=True)
                            # [None, hidden_size]
                        layer_summary(net)

                with tf.variable_scope('logits'):
                    dnn_logit = tf.layers.dense(net, 1)
                    # [None, 1]
                    layer_summary(dnn_logit)
                logits += dnn_logit

    tf.summary.histogram("deep_fm/logits/activations", logits)

    # prediction
    predictions = get_binary_predictions(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    # evaluation
    losses = get_binary_losses(labels, logits)
    metrics = get_binary_metrics(labels, predictions["logistic"], losses["unreduced_loss"])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=losses["loss"],
            eval_metric_ops=metrics
        )

    # training
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(losses["loss"], global_step=tf.train.get_global_step())

    tf.summary.scalar("average_loss", losses["average_loss"])

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=losses["loss"],
            train_op=train_op
        )


def train_main(args):
    # define feature columns
    df = dd.read_csv(args.train_csv, dtype=DATA_DEFAULTS["dtype"]).persist()
    categorical_columns = build_categorical_columns(df, feature_names=DATA_DEFAULTS["feature_names"])

    # clean up model directory
    shutil.rmtree(args.model_dir, ignore_errors=True)
    # define model
    model = tf.estimator.Estimator(
        model_fn,
        args.model_dir,
        params={
            "categorical_columns": categorical_columns,
            "use_linear": not args.exclude_linear,
            "use_mf": not args.exclude_mf,
            "use_dnn": not args.exclude_dnn,
            "embedding_size": args.embedding_size,
            "hidden_units": args.hidden_units,
            "dropout": args.dropout,
        }
    )

    logger.debug("model training started.")
    for n in range(args.num_epochs):
        # train model
        model.train(
            input_fn=lambda: tf_csv_dataset(args.train_csv,
                                            DATA_DEFAULTS["label"],
                                            shuffle=True,
                                            batch_size=args.batch_size)
        )
        # evaluate model
        results = model.evaluate(
            input_fn=lambda: tf_csv_dataset(args.test_csv,
                                            DATA_DEFAULTS["label"],
                                            batch_size=args.batch_size)
        )
        logger.info("epoch %s: %s.", n, results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train-csv", default=str(PROJECT_DIR / DATA_DEFAULTS["train_csv"]),
                        help="path to the training csv data (default: %(default)s)")
    parser.add_argument("--test-csv", default=str(PROJECT_DIR / DATA_DEFAULTS["test_csv"]),
                        help="path to the test csv data (default: %(default)s)")
    parser.add_argument("--model-dir", default="checkpoints/deep_fm",
                        help="model directory (default: %(default)s)")
    parser.add_argument("--exclude-linear", action="store_true",
                        help="flag to exclude linear component (default: %(default)s)")
    parser.add_argument("--exclude-mf", action="store_true",
                        help="flag to exclude mf component (default: %(default)s)")
    parser.add_argument("--exclude-dnn", action="store_true",
                        help="flag to exclude dnn component (default: %(default)s)")
    parser.add_argument("--embedding-size", type=int, default=16,
                        help="embedding size (default: %(default)s)")
    parser.add_argument("--hidden-units", type=int, nargs='+', default=[64, 64, 64],
                        help="hidden layer specification (default: %(default)s)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout rate (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size (default: %(default)s)")
    parser.add_argument("--num-epochs", type=int, default=16,
                        help="number of training epochs (default: %(default)s)")
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
