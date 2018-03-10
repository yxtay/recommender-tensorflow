# Recommendation Models in TensorFlow
Recommendation Models in TensorFlow

## Setup

```bash
# clone repo
git clone git@github.com:yxtay/recommender-tensorflow.git && cd recommender-tensorflow

# create conda environment
conda env create -f=environment.yml

# activate environment
source activate dl
```
## Download Data

```bash
# movielens 100k data: http://files.grouplens.org/datasets/movielens/ml-100k.zip
python -m src.data.ml_100k
```
