import os
from os import listdir
import shutil

DATASET_PATH = "data/raw"
OUTPUT_PATH = "data/dataset"

# Build a dataset
speaker_dirs = [filename for filename in listdir(DATASET_PATH) if filename != ".DS_Store"]
print(speaker_dirs)

speaker_to_files_map = {}
for speaker in speaker_dirs:
    speaker_path = os.path.join(DATASET_PATH, speaker)
    audio_files = [os.path.join(speaker_path, filename) for filename in listdir(speaker_path) if filename.endswith(".WAV")]
    speaker_to_files_map[speaker] = audio_files

print(speaker_to_files_map)

# Copy and rename the files to create the full dataset
for speaker_id, audio_files in speaker_to_files_map.items():
    for audio_file in audio_files:
        source = audio_file
        destination = f'{OUTPUT_PATH}/{speaker_id}_{os.path.basename(audio_file)}'
        dest = shutil.copyfile(source, destination)

print(f'After copying file:\n {listdir(OUTPUT_PATH)}')