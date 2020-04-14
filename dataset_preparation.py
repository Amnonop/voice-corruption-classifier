import os
import shutil
import csv
from pathlib import Path

CSV_PATH = './data/speakers.csv'


def get_directories(parent_dir: Path) -> list:
    return [child_entry for child_entry in parent_dir.iterdir() if child_entry.is_dir()]


def map_speaker_to_files(dataset_filename: str) -> dict:
    dataset_path = Path(dataset_filename)
    speaker_to_file_map = {}

    for entry in get_directories(dataset_path):
        for speaker_dir in get_directories(entry):
            speaker_to_file_map[speaker_dir.name] = [speaker_dir.joinpath(filename) for filename in speaker_dir.iterdir() if filename.suffix == '.WAV']

    return speaker_to_file_map


def write_csv_file(csv_filename: str, speakers_map: dict):
    rows = [["ID", "Sex", "File"]]
    for speaker_dir in speakers_map:
        speaker_sex = speaker_dir[0]
        speaker_id = speaker_dir[1:]
        for speaker_file in speakers_map[speaker_dir]:
            rows.append([speaker_id, speaker_sex, speaker_file])

    csv_path = Path(CSV_PATH)
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

dataset_filename = './data/full/TIMIT/TRAIN'
speakers_map = map_speaker_to_files(dataset_filename)
write_csv_file(CSV_PATH, speakers_map)

dataset_filename = './data/full/TIMIT/TEST'
speakers_map = map_speaker_to_files(dataset_filename)
write_csv_file(CSV_PATH, speakers_map)





