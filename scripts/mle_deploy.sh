#!/bin/usr/env bash
BUCKET="recommender-tensorflow"
REGION="us-central1"

MODEL_NAME="recommender_deep_fm"
MODEL_CHECKPOINT=$(gsutil ls gs://${BUCKET}/checkpoints/ | tail -1)
MODEL_BINARIES=$(gsutil ls ${MODEL_CHECKPOINT}export/exporter/ | tail -1)

gcloud ml-engine models create $MODEL_NAME --regions=$REGION

gcloud ml-engine versions create v1 \
    --model $MODEL_NAME \
    --origin $MODEL_BINARIES \
    --runtime-version 1.10
