from time import strftime, gmtime
from pathlib import Path


def create_results_directories(results_dir: str) -> str:
    results_dir_path = Path.cwd().joinpath(results_dir)
    if not results_dir_path.exists():
        results_dir_path.mkdir()

    current_time = strftime("%d_%m_%Y_%H_%M_%S")
    current_run_path = results_dir_path.joinpath(current_time)
    current_run_path.mkdir()

    snapshot_path = current_run_path.joinpath('snapshot')
    snapshot_path.mkdir()

    best_snapshot_path = current_run_path.joinpath('best_snapshot')
    best_snapshot_path.mkdir()

    return current_run_path