import tensorflow as tf

EVAL_INTERVAL = 60


def get_run_config():
    return tf.estimator.RunConfig(
        save_checkpoints_secs=EVAL_INTERVAL
    )


def get_train_spec(input_fn, train_steps):
    return tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=train_steps
    )


def get_exporter(serving_input_fn):
    return tf.estimator.LatestExporter(
        name="exporter",
        serving_input_receiver_fn=serving_input_fn
    )


def get_eval_spec(input_fn, exporter):
    return tf.estimator.EvalSpec(
        input_fn=input_fn,
        steps=None,  # until OutOfRangeError from input_fn
        exporters=exporter,
        start_delay_secs=min(EVAL_INTERVAL, 120),
        throttle_secs=EVAL_INTERVAL
    )