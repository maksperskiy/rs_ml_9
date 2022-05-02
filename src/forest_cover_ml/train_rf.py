from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score

from .data import get_dataset
from .rf_pipeline import create_pipeline, CRITERION


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="src/forest_cover_ml/data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="src/forest_cover_ml/models/model.joblib",
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
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
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
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    feature_selection: bool,
    n_estimators: int,
    criterion: str,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        if criterion not in CRITERION:
            click.echo(f"Bad criterion: 'entropy' or 'gini'.")
            return

        pipeline = create_pipeline(
            use_scaler=use_scaler, 
            feature_selection=feature_selection,
            n_estimators=n_estimators, 
            criterion=criterion, 
            random_state=random_state
            )

        pipeline.fit(features_train, target_train)

        preds = pipeline.predict(features_val)
        preds_proba = pipeline.predict_proba(features_val)

        accuracy = accuracy_score(target_val, preds)
        roc_auc = roc_auc_score(target_val, preds_proba, multi_class='ovr')
        recall = recall_score(target_val, preds, average="macro")
        
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("recall", recall)
        
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"roc_auc: {roc_auc}.")
        click.echo(f"recall: {recall}.")

        dump(pipeline, save_model_path)

        click.echo(f"Model is saved to {save_model_path}.")