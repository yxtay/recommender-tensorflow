# Recommendation Models in TensorFlow

This repository attempts to implement models for recommendation engines in TensorFlow using the Estimator API with feature columns. 

## Models

- Linear classifer: [`linear.py`](trainers/linear.py)
- DNN classifier: [`deep.py`](trainers/deep.py)
- Linear & DNN classifier: [`linear_deep.py`](trainers/linear_deep.py)
- DeepFM: [`deep_fm.py`](trainers/deep_fm.py)

### DeepFM

TODO: Elaborate on model parameters for DeepFM.

**Usage**
```
python -m trainers.deep_fm -h

usage: deep_fm.py [-h] [--train-csv TRAIN_CSV] [--test-csv TEST_CSV]
                  [--job-dir JOB_DIR] [--restore] [--exclude-linear]
                  [--exclude-mf] [--exclude-dnn]
                  [--embedding-size EMBEDDING_SIZE]
                  [--hidden-units HIDDEN_UNITS [HIDDEN_UNITS ...]]
                  [--dropout DROPOUT] [--batch-size BATCH_SIZE]
                  [--train-steps TRAIN_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --train-csv TRAIN_CSV
                        path to the training csv data (default: data/ml-
                        100k/train.csv)
  --test-csv TEST_CSV   path to the test csv data (default: data/ml-
                        100k/test.csv)
  --job-dir JOB_DIR     job directory (default: checkpoints/deep_fm)
  --restore             whether to restore from job_dir
  --exclude-linear      flag to exclude linear component (default: False)
  --exclude-mf          flag to exclude mf component (default: False)
  --exclude-dnn         flag to exclude dnn component (default: False)
  --embedding-size EMBEDDING_SIZE
                        embedding size (default: 4)
  --hidden-units HIDDEN_UNITS [HIDDEN_UNITS ...]
                        hidden layer specification (default: [16, 16])
  --dropout DROPOUT     dropout rate (default: 0.1)
  --batch-size BATCH_SIZE
                        batch size (default: 32)
  --train-steps TRAIN_STEPS
                        number of training steps (default: 20000)
```

## Setup

```bash
# clone repo
git clone git@github.com:yxtay/recommender-tensorflow.git && cd recommender-tensorflow

# create conda environment
conda env create -f=environment.yml

# activate environment
source activate dl
```
## Download & Process Data

The [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) is used for demonstration purpose. The following script downloads the data, processes and enriches it with a few basic features and serialises it to `csv`.

```bash
# downloads and processes movielens 100k dataset
python -m src.data.ml_100k local
```

**Usage**

```
python -m src.data.ml_100k local -h

usage: ml_100k.py local [-h] [--url URL] [--dest DEST] [--log-path LOG_PATH]

optional arguments:
  -h, --help           show this help message and exit
  --url URL            url of MovieLens 100k data (default:
                       http://files.grouplens.org/datasets/movielens/ml-
                       100k.zip)
  --dest DEST          destination directory for downloaded and extracted
                       files (default: data)
  --log-path LOG_PATH  path of log file (default: main.log)
```

## References

- Harper F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19, 19 pages. DOI=http://dx.doi.org/10.1145/2827872.
- Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... Shah, H. (2016). Wide & Deep Learning for Recommender Systems. arXiv:1606.07792 \[cs.LG\].
- Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. arXiv:1703.04247 \[cs.IR\].
