9_evaluation_selection homework for RS School Machine Learning course.

This work uses [Forest Cover](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

## Usage
This package allows you to train model for classifing forest cover.
1. Clone this repository to your machine.
2. Download [Forest Cover](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. 1. Run train_rf to train random forest classifier with the following command:
```sh
poetry run train_rf -d <path to csv with data> -s <path to save trained model> --random-state <random state> --folds <number of folds in K-fold CV> --use-scaler <true/false to use scaler> --feature_selection <true/false to use feature selection> --n_estimators <number of estimators of forest> --criterion <gini/entropy criterion of learning>
```