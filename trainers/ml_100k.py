import tensorflow as tf

COLUMNS = ("user_id,item_id,rating,timestamp,datetime,year,month,day,week,dayofweek,"
           "age,gender,occupation,zipcode,zipcode1,zipcode2,zipcode3,"
           "title,release,video_release,imdb,unknown,action,adventure,animation,children,"
           "comedy,crime,documentary,drama,fantasy,filmnoir,horror,musical,mystery,romance,"
           "scifi,thriller,war,western,release_date,release_year").split(",")
GENRE = ("unknown,action,adventure,animation,children,comedy,crime,documentary,drama,fantasy,"
         "filmnoir,horror,musical,mystery,romance,scifi,thriller,war,western").split(",")
LABEL_COL = "rating"
DEFAULTS = [[0], [0], [0], [0], ["null"], [0], [0], [0], [0], [0],
            [0], ["null"], ["null"], ["null"], ["null"], ["null"], ["null"],
            ["null"], ["null"], ["null"], ["null"], [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
            [0], [0], [0], [0], ["null"], [0]]


def get_feature_columns(embedding_size=4):
    user_fc = tf.feature_column.categorical_column_with_hash_bucket("user_id", 1000, tf.int32)
    item_fc = tf.feature_column.categorical_column_with_hash_bucket("item_id", 2000, tf.int32)

    # user features
    age_fc = tf.feature_column.numeric_column("age")
    age_buckets = tf.feature_column.bucketized_column(age_fc, list(range(15, 66, 10)))
    gender_fc = tf.feature_column.categorical_column_with_vocabulary_list(
        "gender", ["F", "M"],
        num_oov_buckets=1
    )
    occupation_fc = tf.feature_column.categorical_column_with_hash_bucket("occupation", 50)
    zipcode_fc = tf.feature_column.categorical_column_with_hash_bucket("zipcode", 1000)

    # item features
    release_year_fc = tf.feature_column.numeric_column("release_year")
    release_year_buckets = tf.feature_column.bucketized_column(release_year_fc, list(range(1930, 1991, 10)))
    genre_fc = [tf.feature_column.categorical_column_with_identity(col, 2) for col in GENRE]

    linear_columns = [user_fc, item_fc, age_buckets, gender_fc, occupation_fc, zipcode_fc, release_year_buckets] + genre_fc
    deep_columns = [tf.feature_column.embedding_column(fc, embedding_size) for fc in linear_columns]
    return {"linear": linear_columns, "deep": deep_columns}


def get_input_fn(csv_path, mode=tf.estimator.ModeKeys.TRAIN, batch_size=32, cutoff=5):
    def input_fn():
        def parse_csv(value):
            columns = tf.decode_csv(value, DEFAULTS)
            features = dict(zip(COLUMNS, columns))
            label = features.pop(LABEL_COL)
            label = tf.math.greater_equal(label, cutoff)
            return features, label

        # read, parse, shuffle and batch dataset
        dataset = tf.data.TextLineDataset(csv_path).skip(1)  # skip header
        if mode == tf.estimator.ModeKeys.TRAIN:
            # shuffle and repeat
            dataset = dataset.shuffle(16 * batch_size).repeat()

        dataset = dataset.map(parse_csv, num_parallel_calls=8)
        dataset = dataset.batch(batch_size)
        return dataset

    return input_fn


def serving_input_fn():
    feature_placeholders = {
        "user_id": tf.placeholder(tf.int32, [None]),
        "item_id": tf.placeholder(tf.int32, [None]),

        "age": tf.placeholder(tf.int32, [None]),
        "gender": tf.placeholder(tf.string, [None]),
        "occupation": tf.placeholder(tf.string, [None]),
        "zipcode": tf.placeholder(tf.string, [None]),

        "release_year": tf.placeholder(tf.int32, [None]),
    }
    feature_placeholders.update({
        col: tf.placeholder_with_default(tf.constant([0]), [None]) for col in GENRE
    })

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=feature_placeholders
    )
