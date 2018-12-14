#!/bin/usr/env bash
BUCKET="recommender-tensorflow"
PROJECT="default-223103"
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
