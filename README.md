# Recommendation Models in TensorFlow

This repository attempts to implement models for recommendation engines in TensorFlow using the Estimators API. 

## Models

- Linear classifer: [`linear.py`](src/models/linear.py)
- DNN classifier: [`deep.py`](src/models/deep.py)
- DNN classifier: [`wide_deep.py`](src/models/wide_deep.py)
- Factorisation machine
- DeepFM

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

- Harper F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19, 19 pages. DOI=http://dx.doi.org/10.1145/2827872.
- Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... Shah, H. (2016). Wide & Deep Learning for Recommender Systems. arXiv:1606.07792 \[cs.LG\].
- Guo, H., Tang, R., Ye, Y., Li, Z., He, X. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. arXiv:1703.04247 \[cs.IR\].
