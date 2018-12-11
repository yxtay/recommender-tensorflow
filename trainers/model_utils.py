import tensorflow as tf


def layer_summary(value):
    tf.summary.scalar("fraction_of_zero_values", tf.nn.zero_fraction(value))
    tf.summary.histogram("activation", value)


def get_binary_predictions(logits):
    with tf.name_scope("predictions"):
        logistic = tf.sigmoid(logits)
        class_ids = tf.cast(logistic > 0.5, tf.int32)

    predictions = {
        "logits": logits,
        "logistic": logistic,
        "probabilities": logistic,
        "class_id": class_ids,
    }
    return predictions


def get_binary_losses(labels, predictions):
    with tf.name_scope("losses"):
        labels = tf.expand_dims(labels, -1)
        unreduced_loss = tf.losses.sigmoid_cross_entropy(labels, predictions["logits"],
                                                         reduction=tf.losses.Reduction.NONE)
        average_loss = tf.reduce_mean(unreduced_loss)
        loss = tf.reduce_sum(unreduced_loss)

    losses = {
        "unreduced_loss": unreduced_loss,
        "average_loss": average_loss,
        "loss": loss,
    }
    return losses


def get_binary_metric_ops(labels, predictions, losses):
    with tf.name_scope("metrics"):
        labels = tf.expand_dims(labels, -1)
        average_loss = tf.metrics.mean(losses["unreduced_loss"])
        accuracy = tf.metrics.accuracy(labels, predictions["class_id"], name="accuracy")
        auc = tf.metrics.auc(labels, predictions["probabilities"], name="auc")
        auc_precision_recall = tf.metrics.auc(labels, predictions["probabilities"],
                                              curve="PR", name="auc_precision_recall")

    metrics = {
        "accuracy": accuracy,
        "auc": auc,
        "auc_precision_recall": auc_precision_recall,
        "average_loss": average_loss,
    }
    return metrics


def get_train_op(loss, optimizer_name="Adam", learning_rate=0.001):
    optimizer_classes = {
        "Adagrad": tf.train.AdagradOptimizer,
        "Adam": tf.train.AdamOptimizer,
        "Ftrl": tf.train.FtrlOptimizer,
        "RMSProp": tf.train.RMSPropOptimizer,
        "SGD": tf.train.GradientDescentOptimizer,
    }

    with tf.name_scope("train"):
        optimizer = optimizer_classes[optimizer_name](learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return train_op
