import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# Paths
ITM_RAW_DIR = Path('./GT-ITM-Flute-99')
FILOSAX_RAW_DIR = Path('./filosax')
OUT_DIR = Path('./data/processed/combined_data')

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

def parse_itm_annotation(txt_path):
    notes = []
    with open(txt_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            parts = line.split()
            if len(parts) < 4: continue
                
            try:
                # Attempt 1: Original Format (Onset, Duration, X, Pitch)
                onset = float(parts[0])
                duration = float(parts[1])
                pitch_str = parts[3]
            except ValueError:
                try:
                    # Attempt 2: "Izzy" Format (Index, Filename, Onset, Duration, X, Pitch)
                    # This shifts the indices over by 2 to bypass the prepended text!
                    onset = float(parts[2])
                    duration = float(parts[3])
                    pitch_str = parts[5]
                except (ValueError, IndexError):
                    # If both fail, this is just a text header line. Skip it safely.
                    continue
            
            try:
                notes.append({
                    'onset': onset,
                    'offset': onset + duration,
                    'pitch_midi': note_name_to_midi(pitch_str)
                })
            except Exception:
                # Failsafe for unreadable pitch strings
                pass
                
    return notes

def process_gt_itm():
    print("--- Processing GT-ITM-Flute-99 ---")
    audio_dir = ITM_RAW_DIR / 'audio'
    annot_dir = ITM_RAW_DIR / 'annotations'  # <-- FIX 1: Added the 's'
    
    if not audio_dir.exists() or not annot_dir.exists():
        print(f"[ERROR] Missing audio or annotations folder in {ITM_RAW_DIR}")
        return
    
    # Load all text files into memory once for fast searching
    all_annots = list(annot_dir.glob('*.txt'))
    
    for wav_path in tqdm(list(audio_dir.glob('*.wav'))):
        stem = wav_path.stem
        annot_path = None
        
        # <-- FIX 2: Bulletproof substring matching -->
        for txt in all_annots:
            # Match if the stem is inside the txt name AND it's not an augment
            if stem in txt.name and '_aug_' not in txt.name:
                annot_path = txt
                break
                
        if not annot_path:
            print(f"  [WARN] Missing annotation for {stem}. Skipping.")
            continue
            
        stem_out_dir = OUT_DIR / f"itm_{stem}" 
        stem_out_dir.mkdir(exist_ok=True, parents=True)
        
        with open(stem_out_dir / 'notes.json', 'w') as f:
            json.dump(parse_itm_annotation(annot_path), f)
            
        out_wav = stem_out_dir / 'audio_16k.wav'
        if not out_wav.exists():
            subprocess.run([
                'ffmpeg', '-y', '-i', str(wav_path), 
                '-ac', '1', '-ar', '16000', 
                '-filter:a', 'loudnorm=I=-23:LRA=11:TP=-2', 
                str(out_wav), '-loglevel', 'error'
            ], check=True)

def process_filosax():
    print("\n--- Processing Filosax ---")
    if not FILOSAX_RAW_DIR.exists(): return
    
    for p_dir in FILOSAX_RAW_DIR.iterdir():
        if not p_dir.is_dir() or "Participant" not in p_dir.name: continue
        
        p_id = p_dir.name.split()[-1]
        
        for take_dir in tqdm(list(p_dir.iterdir()), desc=f"Participant {p_id}"):
            if not take_dir.is_dir(): continue
            
            wav_path = take_dir / "Sax.wav"
            json_path = take_dir / "annotations.json"
            
            if not wav_path.exists() or not json_path.exists(): continue
                
            stem_out_dir = OUT_DIR / f"filo_p{p_id}_{take_dir.name}"
            stem_out_dir.mkdir(exist_ok=True, parents=True)
            
            # --- FILOSAX PARSING LOGIC ---
            with open(json_path, 'r') as f:
                raw_json = json.load(f)
                
            filo_notes = [] 
            for note in raw_json.get('notes', []):
                # We use the sonic start/end times ('s_') rather than the theoretical beat times ('a_')
                filo_notes.append({
                    'onset': note['s_start_time'], 
                    'offset': note['s_end_time'], 
                    'pitch_midi': note['midi_pitch']
                })
            
            with open(stem_out_dir / 'notes.json', 'w') as f:
                json.dump(filo_notes, f)
            # -----------------------------
                
            out_wav = stem_out_dir / 'audio_16k.wav'
            if not out_wav.exists():
                subprocess.run(['ffmpeg', '-y', '-i', str(wav_path), '-ac', '1', '-ar', '16000', '-filter:a', 'loudnorm=I=-23:LRA=11:TP=-2', str(out_wav), '-loglevel', 'error'], check=True)

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    process_gt_itm()
    process_filosax()
    print("\n=== PREPROCESSING COMPLETE ===")