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
## Download & Process Data

```bash
# movielens 100k dataset: https://grouplens.org/datasets/movielens/100k/
python -m src.data.ml_100k
```

## References

- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872
