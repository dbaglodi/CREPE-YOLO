import os
import random
import sys
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.utils import get_train_val_test_split

def augment_annotation(txt_path, out_path, new_filename_ref, pitch_steps=0, stretch_rate=1.0):
    """Parses the specific ITM Flute .txt format and mathematically shifts the labels."""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    out_lines = []
    for line in lines:
        parts = line.strip('\n').split('\t')
        
        # If line is empty or malformed, preserve it or skip
        if len(parts) < 8:
            if line.strip() != "":
                out_lines.append(line)
            continue
            
        idx = parts[0]
        # parts[1] is the original filename (e.g., Bra_FirstMay.txt). We replace it.
        
        try:
            # --- 1. Time Shift Math ---
            onset = float(parts[2]) / stretch_rate
            offset = float(parts[3]) / stretch_rate
            duration = float(parts[4]) / stretch_rate
            
            annot_type = parts[5]
            note_str = parts[6]
            freq = float(parts[7])
            
            # --- 2. Pitch Shift Math ---
            if pitch_steps != 0:
                freq = freq * (2 ** (pitch_steps / 12.0))
                try:
                    # Convert 'Eb5' to MIDI number, add steps, convert back to string
                    midi_val = librosa.note_to_midi(note_str)
                    new_note_str = librosa.midi_to_note(midi_val + pitch_steps, unicode=False)
                    # Note: librosa defaults to sharps (D#5 instead of Eb5), 
                    # which is perfectly fine for downstream midi parsing.
                except Exception:
                    new_note_str = note_str # Fallback if note string is weird (e.g., 'None')
            else:
                new_note_str = note_str
                
            # Reconstruct the tab-separated line
            new_line = f"{idx}\t{new_filename_ref}\t{onset:.5f}\t{offset:.5f}\t{duration:.5f}\t{annot_type}\t{new_note_str}\t{freq:.3f}\n"
            out_lines.append(new_line)
            
        except Exception as e:
            # Fallback if a specific line fails to parse
            out_lines.append(line)
        
    with open(out_path, 'w') as f:
        f.writelines(out_lines)


def augment_dataset():
    # --- Configuration ---
    raw_audio_dir = 'GT-ITM-Flute-99/audio'
    annotations_dir = 'GT-ITM-Flute-99/annotations'
    
    # Grab all ORIGINAL stems by checking the audio folder
    all_stems = [f.replace('.wav', '') for f in os.listdir(raw_audio_dir) 
                 if f.endswith('.wav') and '_aug_' not in f]
                 
    # Run the seeded split so we only augment the training set
    train_stems, val_stems, test_stems = get_train_val_test_split(all_stems, train_size=0.64, val_size=0.16, test_size=0.2, seed=42)
    
    print(f"Total Stems Found: {len(all_stems)}")
    print(f"Training Stems to Augment: {len(train_stems)}")
    print(f"Test Stems (PROTECTED): {len(test_stems)}")
    print("-" * 40)
    
    # Augmentation Profiles
    pitch_population = [x for x in range(-6, 7) if x != 0] # [-6, 6] excluding 0
    num_pitch_shifts = 2
    
    stretch_population = [0.85, 0.90, 0.95, 1.05, 1.10, 1.15]
    num_stretch = 1
    
    for stem in tqdm(train_stems, desc="Augmenting Train Data"):
        audio_path = os.path.join(raw_audio_dir, f"{stem}.wav")
        # Prepend the "izzy_GT_" prefix matching your annotation folder convention
        txt_path = os.path.join(annotations_dir, f"izzy_GT_{stem}.txt")
        
        if not os.path.exists(audio_path) or not os.path.exists(txt_path):
            continue
            
        y, sr = librosa.load(audio_path, sr=None)
        
        # --- 1. Apply Pitch Shifts ---
        selected_shifts = random.sample(pitch_population, num_pitch_shifts)
        for steps in selected_shifts:
            new_stem = f"{stem}_aug_pitch_{steps}"
            
            # Save Audio
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            sf.write(os.path.join(raw_audio_dir, f"{new_stem}.wav"), y_shifted, sr)
            
            # Save Annotation
            new_txt_path = os.path.join(annotations_dir, f"izzy_GT_{new_stem}.txt")
            augment_annotation(txt_path, new_txt_path, f"{new_stem}.txt", pitch_steps=steps, stretch_rate=1.0)
            
        # --- 2. Apply Time Stretches ---
        selected_stretches = random.sample(stretch_population, num_stretch)
        for rate in selected_stretches:
            rate_int = int(rate * 100)
            new_stem = f"{stem}_aug_speed_{rate_int}"
            
            # Save Audio
            y_stretched = librosa.effects.time_stretch(y, rate=rate)
            sf.write(os.path.join(raw_audio_dir, f"{new_stem}.wav"), y_stretched, sr)
            
            # Save Annotation
            new_txt_path = os.path.join(annotations_dir, f"izzy_GT_{new_stem}.txt")
            augment_annotation(txt_path, new_txt_path, f"{new_stem}.txt", pitch_steps=0, stretch_rate=rate)

    print("\n✅ Augmentation Complete!")
    print("Next step: Run your 'dataset_preprocessing.py' script to convert the new .txt files into notes.json.")

if __name__ == "__main__":
    augment_dataset()
