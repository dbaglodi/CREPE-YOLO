import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# Update this path to wherever you downloaded the Google Drive folder!
RAW_DATASET_PATH = Path('./GT-ITM-Flute-99')
OUT_DIR = Path('./processed/itm_flute')

NOTE_MAP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

def note_name_to_midi(name):
    """Converts a string like 'C#5' or 'Bb4' to a MIDI integer."""
    name = name.strip()
    i = len(name) - 1
    while i >= 0 and (name[i].isdigit() or name[i] == '-'): i -= 1
    note_part = name[:i+1]
    octave = int(name[i+1:]) if name[i+1:] else 4
    semitone = NOTE_MAP[note_part[0].upper()]
    if '#' in note_part: semitone += 1
    elif 'b' in note_part: semitone -= 1
    return 12 * (octave + 1) + semitone

def parse_annotation(txt_path):
    """Parses the 8-column tab-separated ITM-Flute-99 annotations."""
    notes = []
    with open(txt_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if len(parts) < 7: continue
            try:
                onset = float(parts[2])
                offset = float(parts[3])
                p_raw = parts[6].strip()
                pitch_midi = note_name_to_midi(p_raw)
                
                # Filter out malformed or out-of-range notes
                if 21 <= pitch_midi <= 108 and offset > onset:
                    notes.append({'onset': onset, 'offset': offset, 'pitch_midi': pitch_midi})
            except Exception:
                continue
    return sorted(notes, key=lambda x: x['onset'])

def process_dataset():
    print("--- Starting Dataset Preprocessing ---")
    audio_dir = RAW_DATASET_PATH / 'audio'
    annot_dir = RAW_DATASET_PATH / 'annotations'
    
    if not audio_dir.exists() or not annot_dir.exists():
        raise FileNotFoundError(f"Ensure {audio_dir} and {annot_dir} exist.")
        
    audio_files = list(audio_dir.glob('*.wav'))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(audio_files)} audio files. Processing...")
    
    for wav_path in tqdm(audio_files):
        stem = wav_path.stem
        
        # Match annotation file (stripping prefixes like 'izzy_GT_')
        annot_path = None
        for prefix in ['izzy_GT_', 'GT_', 'izzy_']:
            potential_annot = annot_dir / f"{prefix}{stem}.txt"
            if potential_annot.exists():
                annot_path = potential_annot
                break
        
        if not annot_path:
            # Fallback exact match
            potential_annot = annot_dir / f"{stem}.txt"
            if potential_annot.exists():
                annot_path = potential_annot
                
        if not annot_path:
            print(f"  [WARN] Missing annotation for {stem}. Skipping.")
            continue
            
        # Create output directory for this stem
        stem_out_dir = OUT_DIR / stem
        stem_out_dir.mkdir(exist_ok=True)
        
        # 1. Parse and save JSON
        notes = parse_annotation(annot_path)
        with open(stem_out_dir / 'notes.json', 'w') as f:
            json.dump(notes, f)
            
        # 2. Resample and normalize audio using FFmpeg
        out_wav = stem_out_dir / 'audio_16k.wav'
        if not out_wav.exists():
            subprocess.run([
                'ffmpeg', '-y', '-i', str(wav_path),
                '-ac', '1', '-ar', '16000', # 16kHz mono
                '-filter:a', 'loudnorm=I=-23:LRA=11:TP=-2', # EBU R128 standard loudness
                str(out_wav), '-loglevel', 'error'
            ], check=True)

    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"Data is ready in {OUT_DIR}")
    print("Next step: Run 'python precompute_features.py' to generate CREPE tensors.")

if __name__ == "__main__":
    process_dataset()