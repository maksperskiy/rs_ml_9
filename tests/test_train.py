from click.testing import CliRunner
import pytest

from src.forest_cover_ml.train_rf import train as train_rf
from src.forest_cover_ml.train_log import train as train_log


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_criterion(
    runner: CliRunner
) -> None:
    result = runner.invoke(
        train_rf,
        [
            "--criterion",
            "criterion",
        ],
    )
    assert result.exit_code == 0
    assert "Bad criterion" in result.output

def test_error_for_invalid_folds(
    runner: CliRunner
) -> None:
    result = runner.invoke(
        train_log,
        [
            "--folds",
            0
        ]
    )
    assert result.exit_code == 2
    assert "Invalid value for '--folds" in result.output
