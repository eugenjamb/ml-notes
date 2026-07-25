# Machine Learning Notes and Examples

This repository is my personal learning workspace for machine learning, deep
learning, and natural language processing. I use it to turn concepts I study
into small, runnable Python examples that I can revisit later.

The files are written as code notes rather than as a reusable Python package.
They include detailed comments explaining what the main methods do, why they
are used, and how important parameters affect a model.

## Topics

- `supervised/` contains supervised learning algorithms such as regression,
  classification, decision trees, K-nearest neighbours, SVMs, and Naive Bayes.
- `unsupervised/` contains unsupervised learning examples such as clustering
  and dimensionality reduction.
- `ensembled/` contains bagging, boosting, random forest, and stacking examples.
- `regularization_and_tuning/` covers L1, L2, ElasticNet, bias versus variance,
  and hyperparameter search.
- `pipeline/` demonstrates preprocessing pipelines, column transformers, and
  model selection with grid search.
- `deep_learning/` contains TensorFlow/Keras examples for artificial neural
  networks and convolutional neural networks.
- `nlp/` currently contains text-preprocessing examples.
- `data/` contains small local datasets used by some examples.

## Running an Example

Create and activate a virtual environment from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the libraries needed by the example you want to run. Most traditional
machine-learning examples use:

```powershell
python -m pip install numpy pandas matplotlib seaborn scikit-learn
```

Deep-learning examples may additionally require:

```powershell
python -m pip install tensorflow keras keras-tuner tensorboard pillow
```

Then run an individual file from the repository root, for example:

```powershell
python supervised\knn\kneighbour_classifier_example.py
```

Some examples train several models or display graphs, so their running time and
output vary. This repository is primarily my study notebook: examples may be
expanded or reorganized as I learn new topics.
