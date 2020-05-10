from time import strftime, gmtime
from pathlib import Path


def create_results_directories(results_dir: str) -> str:
    results_dir_path = Path.cwd().joinpath(results_dir)
    if not results_dir_path.exists():
        print(f'Could not find directory for results. Creating {results_dir_path}')
        results_dir_path.mkdir()

    current_time = strftime("%d_%m_%Y_%H_%M_%S")
    current_run_path = results_dir_path.joinpath(current_time)
    print(f'Creating directory for current run at {current_run_path}')
    current_run_path.mkdir()

    snapshot_path = current_run_path.joinpath('snapshot')
    print(f'Creating directory for snapshots at {snapshot_path}')
    snapshot_path.mkdir()

    best_snapshot_path = current_run_path.joinpath('best_snapshot')
    print(f'Creating directory for best snapshot at {best_snapshot_path}')
    best_snapshot_path.mkdir()

    return current_run_path