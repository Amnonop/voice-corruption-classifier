from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from commons import *
from configuration import Configuration
from utils import create_results_directories
from dataset_transforms import TransformsComposer, ToTensor, Rescale
from siamese_loader import SiameseLoader
from siamese import Siamese

DATA_DIR = './data'

def main():
    config_filename = Path.cwd().joinpath(CONFIGS_DIR).joinpath(CONFIG_FILENAME)
    config = Configuration(config_filename)

    data_dir_path = Path.cwd().joinpath(DATA_DIR)

    results_dir_path = Path.cwd().joinpath(RESULTS_DIR)
    current_run_path = create_results_directories(results_dir_path)

    transforms = TransformsComposer([Rescale(output_size=SAMPLE_SIZE), ToTensor()])

    encoder = LabelEncoder()

    # Load data
    data_loader = SiameseLoader(data_dir_path, transforms)

    batch = data_loader.get_batch(4)
    one_shot = data_loader.make_oneshot_task(20)

    model = Siamese()


if __name__ == '__main__':
    main()
