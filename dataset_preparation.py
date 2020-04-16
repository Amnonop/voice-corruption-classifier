import os
import shutil
import csv
from pathlib import Path

CSV_PATH = './data/speakers.csv'


class SpeakerData:
    def __init__(self, id: str, sex: str, dialect: str, files: list):
        self.id = id
        self.sex = sex
        self.dialect = dialect
        self.files = files


def get_directories(parent_dir: Path) -> list:
    return [child_entry for child_entry in parent_dir.iterdir() if child_entry.is_dir()]


def create_speakers_data(dataset_filename: str) -> list:
    dataset_path = Path(dataset_filename)
    speakers = []

    for dialect_dir in get_directories(dataset_path):
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
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

dataset_filename = './data/full/TIMIT/TRAIN'
speakers = create_speakers_data(dataset_filename)
write_csv_file(CSV_PATH, speakers)

dataset_filename = './data/full/TIMIT/TEST'
speakers = create_speakers_data(dataset_filename)
write_csv_file(CSV_PATH, speakers)





