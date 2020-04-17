import os
import shutil
import csv
from pathlib import Path

from commons import CONFIGS_DIR
from configuration import load_config

CONFIG_FILENAME = 'prepare_config.json'
CSV_FILENAME = 'speakers.csv'


class SpeakerData:
    def __init__(self, id: str, sex: str, dialect: str, files: list):
        self.id = id
        self.sex = sex
        self.dialect = dialect
        self.files = files


def get_directories(parent_dir: Path) -> list:
    return [child_entry for child_entry in parent_dir.iterdir() if child_entry.is_dir()]


def get_speakers_data(dataset_dir: str) -> list:
    dataset_path = Path(dataset_dir)
    speakers = []

    for mode_dir in get_directories(dataset_path):
        for dialect_dir in get_directories(mode_dir):
            for speaker_dir in get_directories(dialect_dir):
                speaker_sex = speaker_dir.name[0]
                speaker_id = speaker_dir.name[1:]
                speaker_files = [filename for filename in speaker_dir.iterdir() if filename.suffix == '.WAV']
                speaker = SpeakerData(speaker_id, speaker_sex, dialect_dir.name, speaker_files)
                speakers.append(speaker)

    return speakers


def write_csv_file(csv_filename: str, speakers: list):
    rows = [["ID", "Sex", "Dialect", "File"]]
    for speaker in speakers:
        for speaker_file in speaker.files:
            rows.append([speaker.id, speaker.sex, speaker.dialect, speaker_file])

    csv_path = Path(csv_filename)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def main():
    config_path = Path(CONFIGS_DIR).joinpath(CONFIG_FILENAME)
    config = load_config(config_path)

    dataset_dir = config['dataset_dir']
    speakers = get_speakers_data(dataset_dir)

    csv_dir = config['csv_path']
    csv_path = Path(csv_dir).joinpath(CSV_FILENAME)
    write_csv_file(csv_path, speakers)


if __name__ == '__main__':
    main()





