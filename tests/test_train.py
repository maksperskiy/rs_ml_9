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
    assert result.exit_code == 2
    assert "Invalid value" in result.output

def test_success_for_invalid_criterion(
    runner: CliRunner
) -> None:
    result = runner.invoke(
        train_rf,
        [
            "--criterion",
            "gini",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--criterion'" not in result.output


def test_error_for_invalid_logreg_c(
    runner: CliRunner
) -> None:
    result = runner.invoke(
        train_log,
        [
            "--logreg-c",
            1.2,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value" in result.output

def test_success_for_invalid_logreg_c(
    runner: CliRunner
) -> None:
    result = runner.invoke(
        train_log,
        [
            "--logreg-c",
            0.1,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--logreg-c'" not in result.output