from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate

from .data import get_dataset, get_X_y
from .log_pipeline import create_pipeline


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
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
def train(
    train_dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    folds: int,
    use_scaler: bool,
    feature_selection: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    X, y = get_X_y(train_dataset_path, random_state)
    with mlflow.start_run(run_name="log_model"):
        pipeline = create_pipeline(
            use_scaler=use_scaler,
            feature_selection=feature_selection,
            max_iter=max_iter,
            logreg_C=logreg_c,
            random_state=random_state,
        )

        scoring = ["accuracy", "recall_macro", "roc_auc_ovr"]

        scores = cross_validate(
            pipeline, X, y, scoring=scoring, cv=folds, return_train_score=False
        )

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)

        for name in scores.keys():
            print("%s: %.4f" % (name, np.average(scores[name])))
            mlflow.log_metric(name, np.average(scores[name]))
            click.echo(f"{name}: {np.average(scores[name])}.")

        dump(pipeline, save_model_path)

        click.echo(f"Model is saved to {save_model_path}.")
