import os
import shutil
import csv
from pathlib import Path

from commons import CONFIGS_DIR
from configuration import load_config

CONFIG_FILENAME = 'prepare_config.json'
CSV_FILENAME_PREFIX = 'siamese_speakers_{0}.csv'


class SpeakerData:
    def __init__(self, id: str, sex: str, dialect: str, files: list):
        self.id = id
        self.sex = sex
        self.dialect = dialect
        self.files = files


def get_directories(parent_dir: Path) -> list:
    return [child_entry for child_entry in parent_dir.iterdir() if child_entry.is_dir()]


def get_speakers_data(dataset_dir: str) -> dict:
    dataset_path = Path.cwd().joinpath(dataset_dir)
    speakers = {
        'train': [],
        'test': []
    }

    for mode_dir in get_directories(dataset_path):
        mode = mode_dir.name.lower()
        for dialect_dir in get_directories(mode_dir):
            for speaker_dir in get_directories(dialect_dir):
                speaker_sex = speaker_dir.name[0]
                speaker_id = speaker_dir.name[1:]
                speaker_files = [filename for filename in speaker_dir.iterdir() if filename.suffix == '.WAV']
                speaker = SpeakerData(speaker_id, speaker_sex, dialect_dir.name, speaker_files)
                speakers[mode].append(speaker)

    return speakers


def write_csv_file(csv_dir: str, csv_filename: str, speakers: list):
    rows = [["ID", "Sex", "Dialect", "File"]]
    for speaker in speakers:
        for speaker_file in speaker.files:
            rows.append([speaker.id, speaker.sex, speaker.dialect, speaker_file])

    csv_path = Path.cwd().joinpath(csv_dir).joinpath(csv_filename)

    print(f'Writing speakers data to {csv_filename}')
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def write_files(csv_filename_prefix: str, csv_dir:str, speakers: dict):
    for mode in speakers.keys():
        csv_filename = csv_filename_prefix.format(mode)
        write_csv_file(csv_dir, csv_filename, speakers[mode])


def main():
    config_path = Path.cwd().joinpath(CONFIGS_DIR).joinpath(CONFIG_FILENAME)
    config = load_config(config_path)

    dataset_dir = config['dataset_dir']

    print(f'Getting speakers data from {dataset_dir}')
    speakers = get_speakers_data(dataset_dir)

    csv_dir = config['csv_path']

    write_files(CSV_FILENAME_PREFIX, csv_dir, speakers)

    print('Done')


if __name__ == '__main__':
    main()





