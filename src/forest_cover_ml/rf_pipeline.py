from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import enum

CRITERION = [
    "gini",
    "entropy"
]

def create_pipeline(
    use_scaler: bool, feature_selection: bool, n_estimators: int, criterion: str, random_state: int
) -> Pipeline:
    pipeline_steps = []

    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if feature_selection:
        pipeline_steps.append(
            (
                'feature_selection',
                SelectFromModel(
                    LinearSVC(
                        random_state=random_state, 
                        penalty="l2"
                    )
                )
            ))

    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                random_state=random_state
            ),
        )
    )

    return Pipeline(steps=pipeline_steps)
