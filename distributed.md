# Distributed Model Training and Evaluation on Google Cloud Platform

The trainer module in this repository also allows for distributed model training and evaluation on Google Cloud Platform.

## Setup

```bash
# clone repo
git clone git@github.com:yxtay/recommender-tensorflow.git && cd recommender-tensorflow

# create conda environment
conda env create -f=environment.yml

# activate environment
source activate dl
```

## Google Cloud Platform Credentials

TODO: Add instructions to set up `gcloud` and create credentials file.

## Download & Process Data

- Downloads and unzips data locally
- Uploads data to Bigquery
- Processes data on Bigquery
- Exports data as CSV to Google Cloud Storage

```bash
# downloads and processes movielens 100k dataset
python -m src.data.ml_100k gcp
```

**Usage**

```bash
python -m src.data.ml_100k gcp -h

usage: ml_100k.py gcp [-h] [--url URL] [--dest DEST] [--dataset DATASET]
                      [--gcs-bucket GCS_BUCKET] [--credentials CREDENTIALS]
                      [--log-path LOG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --url URL             url of MovieLens 100k data (default: http://files.grou
                        plens.org/datasets/movielens/ml-100k.zip)
  --dest DEST           destination directory for downloaded and extracted
                        files (default: data)
  --dataset DATASET     dataset name to save datatables
  --gcs-bucket GCS_BUCKET
                        google cloud storage bucket to store processed files
  --credentials CREDENTIALS
                        json file containing google cloud credentials
  --log-path LOG_PATH   path of log file (default: main.log)

```

## Distributed Training & Evaluation of DeepFM

```bash
# change accordingly
BUCKET="recommender-tensorflow"
REGION="us-central1"

MODEL_TYPE="deep_fm"
JOB_NAME="${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)"
OUTPUT_PATH="gs://${BUCKET}/checkpoints/$JOB_NAME"
PACKAGE_PATH="trainers/"
TRAIN_DATA="gs://${BUCKET}/data/ml-100k/train.csv"
TEST_DATA="gs://${BUCKET}/data/ml-100k/test.csv"

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.10 \
    --module-name trainers.$MODEL_TYPE \
    --package-path $PACKAGE_PATH \
    --region $REGION \
    -- \
    --train-csv $TRAIN_DATA \
    --test-csv $TEST_DATA \
    --train-steps 100000
```

## Tensorboard

You may inspect model training metrics with Tensorboard

```bash
BUCKET="recommender-tensorflow"
OUTPUT_PATH="gs://${BUCKET}/checkpoints/"

tensorboard --logdir OUTPUT_PATH
```
