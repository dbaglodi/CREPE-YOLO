import sys
import os
import subprocess
from pathlib import Path
import functools

# Force Python to print to the terminal instantly before C-level crashes
print = functools.partial(print, flush=True)

# ==============================================================================
# PHASE 2: ISOLATED PYTORCH CONVERSION PROCESS
# ==============================================================================
def run_pytorch_conversion():
    print("\n>>> [PHASE 2] Process spawned successfully.")
    print(">>> [PHASE 2] Importing PyTorch...")
    import torch
    import numpy as np
    from tqdm import tqdm
    
    processed_dir = Path('./data/processed/itm_flute')
    stems = [d.name for d in processed_dir.iterdir() if d.is_dir()]
    downsample_factor = 32
    
    print(f">>> [PHASE 2] Found {len(stems)} stems to check for conversion.")
    
    for stem in tqdm(stems, desc="Converting to Tensors"):
        stem_dir = processed_dir / stem
        npz_path = stem_dir / 'features.npz'
        pt_path = stem_dir / 'features.pt'
        
        if not npz_path.exists() or pt_path.exists():
            continue
            
        data = np.load(npz_path)
        features_np = {k: data[k] for k in data.files}
        features_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in features_np.items()}
        
        if features_t['gradient'].max() > 0:
            features_t['gradient'] = features_t['gradient'] / features_t['gradient'].max()
            
        T = features_t['posteriorgram'].shape[-1]
        remainder = T % downsample_factor
        if remainder != 0:
            pad_len = downsample_factor - remainder
            features_t = {k: torch.nn.functional.pad(v, (0, pad_len)) for k, v in features_t.items()}
            
        torch.save(features_t, pt_path)
        npz_path.unlink() 
        
    print(">>> [PHASE 2] Conversion loop finished!")
    print(">>> [PHASE 2] Triggering hard exit to prevent PyTorch malloc crash...")
    os._exit(0) # Nuke the PyTorch process

if len(sys.argv) > 1 and sys.argv[1] == '--convert':
    run_pytorch_conversion()
    # Should never reach here due to os._exit(0) above

# ==============================================================================
# PHASE 1: TENSORFLOW EXTRACTION PROCESS
# ==============================================================================
print("\n>>> [PHASE 1] Script started. Setting environment variables...")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'

print(">>> [PHASE 1] Importing TensorFlow...")
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
    print(">>> [PHASE 1] Apple Silicon GPU successfully disabled for stability.")
except Exception as e:
    print(f">>> [PHASE 1] GPU disable failed: {e}")

print(">>> [PHASE 1] Importing Librosa and CREPE...")
import numpy as np
import librosa
import crepe
from keras.models import Model
from tqdm import tqdm

class CREPEFeatureExtractor:
    def __init__(self, model_capacity='full', step_size=10):
        print(">>> [PHASE 1] Building CREPE Model in Keras...")
        self.step_size = step_size
        self.sr = 16000 
        self.base_model = crepe.core.build_and_load_model(model_capacity)
        embedding_layer = self.base_model.layers[-2].output
        classification_layer = self.base_model.layers[-1].output
        self.feature_model = Model(
            inputs=self.base_model.input, 
            outputs=[classification_layer, embedding_layer]
        )
        print(">>> [PHASE 1] CREPE Model built successfully.")

    def extract_features_numpy(self, audio_path: str) -> dict:
        audio, sr = librosa.load(audio_path, sr=self.sr)
        audio = np.pad(audio, 512, mode='constant', constant_values=0)
        
        hop_length = int(self.sr * self.step_size / 1000)
        n_frames = 1 + int((len(audio) - 1024) / hop_length)
        
        frames = np.zeros((n_frames, 1024), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            frames[i] = audio[start : start + 1024]
            
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.maximum(np.std(frames, axis=1)[:, np.newaxis], 1e-8)
        
        activations = []
        embeddings = []
        batch_size = 128
        
        for i in range(0, n_frames, batch_size):
            batch = frames[i : i + batch_size]
            act, emb = self.feature_model.predict_on_batch(batch)
            activations.append(np.array(act))
            embeddings.append(np.array(emb))
            
        activation = np.concatenate(activations, axis=0)
        embedding = np.concatenate(embeddings, axis=0)
        confidence = np.max(activation, axis=1)
        cents = crepe.core.to_local_average_cents(activation)
        frequency = 10 * 2 ** (cents / 1200)
        gradient = np.abs(np.gradient(frequency))
        
        return {
            'posteriorgram': activation.T[np.newaxis, :, :],       
            'embedding': embedding.T[np.newaxis, :, :],            
            'confidence': confidence.reshape(1, 1, -1),            
            'gradient': gradient.reshape(1, 1, -1)                 
        }

def main():
    print(">>> [PHASE 1] Entering Main Loop...")
    processed_dir = Path('./data/processed/itm_flute')
    raw_audio_dir = Path('GT-ITM-Flute-99/audio')
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Directory {processed_dir} not found.")
        
    stems = [d.name for d in processed_dir.iterdir() if d.is_dir()]
    print(f">>> [PHASE 1] Found {len(stems)} stem directories.")
    
    extractor = CREPEFeatureExtractor(model_capacity='full', step_size=10)
    
    for stem in tqdm(stems, desc="Extracting Audio Features"):
        stem_dir = processed_dir / stem
        npz_path = stem_dir / 'features.npz'
        pt_path = stem_dir / 'features.pt'
        
        audio_path_processed = stem_dir / 'audio_16k.wav'
        audio_path_raw = raw_audio_dir / f"{stem}.wav"
        
        if pt_path.exists() or npz_path.exists():
            continue
            
        if audio_path_processed.exists():
            target_audio = audio_path_processed
        elif audio_path_raw.exists():
            target_audio = audio_path_raw
        else:
            continue
            
        try:
            features_np = extractor.extract_features_numpy(str(target_audio))
            np.savez(npz_path, **features_np)
        except Exception as e:
            print(f"\n[ERROR] Failed on {stem}: {e}")

    print("\n>>> [PHASE 1] TF Extraction finished. Spawning PyTorch subprocess...")
    subprocess.run([sys.executable, __file__, '--convert'], check=True)
    
    print(">>> [PHASE 1] PyTorch subprocess complete. Triggering hard exit...")
    os._exit(0)

if __name__ == "__main__":
    main()
