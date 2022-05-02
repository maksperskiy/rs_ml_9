from sklearn import feature_selection
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC


def create_pipeline(
    use_scaler: bool, feature_selection: bool, max_iter: int, logreg_C: float, random_state: int
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
                        penalty="l1"
                    )
                )
            ))

    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )

    return Pipeline(steps=pipeline_steps)
