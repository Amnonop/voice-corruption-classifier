import os
from os import listdir
import shutil
import csv

RAW_DATASET = "data/raw"
OUTPUT_DIR = "data/clean_speech"
INFO_FILENAME = "spkrinfo.csv"

speaker_dirs = [directory_name for directory_name in listdir(RAW_DATASET) if directory_name != ".DS_Store"]

speaker_to_audio_files_map = {}
for speaker_id in speaker_dirs:
    speaker_dir = os.path.join(RAW_DATASET, speaker_id)
    speaker_to_audio_files_map[speaker_id] = [os.path.join(speaker_dir, filename) for filename in listdir(speaker_dir) if filename.endswith(".WAV")]

for speaker_id, audio_files in speaker_to_audio_files_map.items():
    for audio_file in audio_files:
        source = audio_file
        destination = os.path.join(OUTPUT_DIR, f'{speaker_id}_{os.path.basename(audio_file)}')
        dest = shutil.copyfile(source, destination)

print(listdir(OUTPUT_DIR))

# Write the csv file for labeling
speaker_files = [filename for filename in listdir(OUTPUT_DIR) if filename != ".DS_Store"]
rows = [["ID", "Sex", "File"]]
for speaker_file in speaker_files:
    filename = os.path.splitext(speaker_file)[0]
    speaker_sex = filename[0]
    speaker_id = filename[1:5]
    rows.append([speaker_id, speaker_sex, speaker_file])

speaker_info_filepath = os.path.join(OUTPUT_DIR, INFO_FILENAME)
with open(speaker_info_filepath, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)




