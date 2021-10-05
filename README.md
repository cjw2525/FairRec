# Software for equal experience in recommender systems. 

Run the following command to install required packages:
```bash
pip install -r requirements.txt
```
Example ipython notebook file for running our algorithm is provided in 'experiments.ipynb'.
For those who want to reproduce our results, please visit the following websites to download real datasets:
- MovieLens 1M dataset: http://www.movielens.org/
- Last FM 360K dataset: http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html

We tested the code on a number of different machines that have Python 3.7.4 or 3.7.5 with Ubuntu 16.04 OS.

# Folder structure:
<pre>
.
├── data 
│   ├── last-fm       # download real datasets at these folders. 
│   └── ml-1m
├── models
│   ├── autoencoder.py
│   └── matrix_factorization.py
├── preprocessing
│   ├── lastfm.py
│   ├── movielens.py
│   └── synthetic.py
├── results
│   ├── lastfm
│   ├── movielens
│   └── synthetic
├── utils
│   ├── metrics.py
│   └── regularizers.py
├── train.py
├── requirements.txt
└── experiments.ipynb
</pre>