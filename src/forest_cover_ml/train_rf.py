from io import BytesIO
from pathlib import Path
from joblib import dump, load
import numpy as np

import click
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import cross_validate

from .data import get_X_y, get_dataset
from .rf_pipeline import create_pipeline, CRITERION


@click.command()
@click.option(
    "-d",
    "--train-dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="models/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--folds",
    default=5,
    type=click.IntRange(1, 10, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--feature_selection",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--n_estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default="entropy",
    type=str,
    show_default=True,
)
def train(
    train_dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    folds: int,
    use_scaler: bool,
    feature_selection: bool,
    n_estimators: int,
    criterion: str,
) -> None:
    if criterion not in CRITERION:
        click.echo(f"Bad criterion: 'entropy' or 'gini'.")
        return

    X, y = get_X_y(train_dataset_path, random_state)
    with mlflow.start_run(run_name="rf_model"):
        pipeline = create_pipeline(
            use_scaler=use_scaler,
            feature_selection=feature_selection,
            n_estimators=n_estimators,
            criterion=criterion,
            random_state=random_state,
        )

        scoring = ["accuracy", "recall_macro", "roc_auc_ovr"]

        scores = cross_validate(
            pipeline, X, y, scoring=scoring, cv=folds, return_train_score=False
        )

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)

        for name in scores.keys():
            print("%s: %.4f" % (name, np.average(scores[name])))
            mlflow.log_metric(name, np.average(scores[name]))
            click.echo(f"{name}: {np.average(scores[name])}.")

        features_train, features_val, target_train, target_val = get_dataset(
            train_dataset_path,
            random_state,
            1/folds,
        )
        pipeline.fit(features_train, target_train)
        dump(pipeline, save_model_path)

        click.echo(f"Model is saved to {save_model_path}.")


@click.command()
@click.option(
    "-d",
    "--test-dataset-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-m",
    "--model-path",
    default="models/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def test_on_last_model(
    test_dataset_path: Path,
    model_path: Path
) -> None:
    X = pd.read_csv(test_dataset_path)
    
    model = load(model_path)
    
    y_pred = model.predict(X)
    
    predictions_test = pd.DataFrame(y_pred)
    predictions_test.columns = ['Cover_Type']

    result = pd.concat([X["Id"], predictions_test], axis=1)

    result.to_csv('data/results.csv', index=False)