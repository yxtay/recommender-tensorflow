# Recommendation Models in TensorFlow

This repository attempts to implement models for recommendation engines in TensorFlow using the Estimators API with feature columns. 

## Models

- Linear classifer: [`linear.py`](src/models/linear.py)
- DNN classifier: [`deep.py`](src/models/deep.py)
- DNN classifier: [`wide_deep.py`](src/models/wide_deep.py)
- Factorisation machine
- DeepFM: [`deep_fm.py`](src/models/deep_fm.py)

### DeepFM

TODO: Elaborate on model parameters for DeepFM.

**Usage**
```
python -m src.models.deep_fm -h

usage: deep_fm.py [-h] [--train-csv TRAIN_CSV] [--test-csv TEST_CSV]
                  [--model-dir MODEL_DIR] [--linear] [--fm] [--dnn]
                  [--embedding-size EMBEDDING_SIZE]
                  [--hidden-units HIDDEN_UNITS [HIDDEN_UNITS ...]]
                  [--dropout DROPOUT] [--batch-size BATCH_SIZE]
                  [--num-epochs NUM_EPOCHS] [--log-path LOG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --train-csv TRAIN_CSV
                        path to the training csv data (default:
                        data/ml-100k/train.csv)
  --test-csv TEST_CSV   path to the test csv data (default:
                        data/ml-100k/test.csv)
  --model-dir MODEL_DIR
                        model directory (default: checkpoints/deep_fm)
  --linear              flag to exclude linear component (default: True)
  --fm                  flag to exclude fm component (default: True)
  --dnn                 flag to exclude dnn component (default: True)
  --embedding-size EMBEDDING_SIZE
                        embedding size (default: 16)
  --hidden-units HIDDEN_UNITS [HIDDEN_UNITS ...]
                        hidden layer specification (default: [64, 64, 64])
  --dropout DROPOUT     dropout rate (default: 0.1)
  --batch-size BATCH_SIZE
                        batch size (default: 32)
  --num-epochs NUM_EPOCHS
                        number of training epochs (default: 16)
  --log-path LOG_PATH   path of log file (default:
                        main.log)
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

The [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) is used for demonstration purpose. The following script downloads the data, processes and enriches it with a few basic features and serialises it to `csv` and `tfrecords`.

```bash
# downloads and processes movielens 100k dataset
python -m src.data.ml_100k
```

**Usage**

```
python -m src.data.ml_100k

usage: ml_100k.py [-h] [--url URL] [--dest DEST] [--log-path LOG_PATH]

Download, extract and prepare MovieLens 100k data.

optional arguments:
  -h, --help           show this help message and exit
  --url URL            url of MovieLens 100k data (default: http://files.group
                       lens.org/datasets/movielens/ml-100k.zip)
  --dest DEST          destination directory for downloaded and extracted
                       files (default: data)
  --log-path LOG_PATH  path of log file (default:
                       main.log)
```

## References

- Harper F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19, 19 pages. DOI=http://dx.doi.org/10.1145/2827872.
- Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... Shah, H. (2016). Wide & Deep Learning for Recommender Systems. arXiv:1606.07792 \[cs.LG\].
- Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. arXiv:1703.04247 \[cs.IR\].
