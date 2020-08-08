from pathlib import Path

class DatasetNShot:
    def __init__(self, root: str, batch_size: int, n_way: int, k_shot: int, k_query: int, sample_size: int, device=None):
        self.resize = sample_size
        self.device = device
        if not Path(root).joinpath("").is_file():