from pathlib import Path
import pandas as pd
import pandas_profiling

import click


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-path",
    default="pandas_profiling/report.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def profile(
    dataset_path: Path,
    save_path: Path
) -> None:
    df = pd.read_csv(dataset_path)

    profile = pandas_profiling.ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file(f"{save_path}")
    
    click.echo(f"Pandas profile is saved to {save_path}.")