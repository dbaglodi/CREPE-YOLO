import sys
import os
import subprocess
from pathlib import Path

# ==============================================================================
# PHASE 2: ISOLATED PYTORCH CONVERSION PROCESS
# ==============================================================================
def run_pytorch_conversion():
    """Runs in a completely separate OS process to prevent memory collisions."""
    import torch
    import numpy as np
    from tqdm import tqdm
    
    processed_dir = Path('./processed/itm_flute')
    stems = [d.name for d in processed_dir.iterdir() if d.is_dir()]
    
    downsample_factor = 32
    
    for stem in tqdm(stems, desc="Converting to PyTorch Tensors"):
        stem_dir = processed_dir / stem
        npz_path = stem_dir / 'features.npz'
        pt_path = stem_dir / 'features.pt'
        
        if not npz_path.exists() or pt_path.exists():
            continue
            
        # Load raw numpy arrays
        data = np.load(npz_path)
        features_np = {k: data[k] for k in data.files}
        
        # Convert to Tensors
        features_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in features_np.items()}
        
        # Normalize gradient
        if features_t['gradient'].max() > 0:
            features_t['gradient'] = features_t['gradient'] / features_t['gradient'].max()
            
        # Pad Time Dimension for YOLO
        T = features_t['posteriorgram'].shape[-1]
        remainder = T % downsample_factor
        if remainder != 0:
            pad_len = downsample_factor - remainder
            features_t = {k: torch.nn.functional.pad(v, (0, pad_len)) for k, v in features_t.items()}
            
        # Save and cleanup
        torch.save(features_t, pt_path)
        npz_path.unlink() # Delete the temporary numpy file
        
    print("\n=== ALL FEATURES PRE-COMPUTED AND SAVED ===")

# Check if this script was called as the subprocess
if len(sys.argv) > 1 and sys.argv[1] == '--convert':
    run_pytorch_conversion()
    sys.exit(0)

# ==============================================================================
# PHASE 1: TENSORFLOW EXTRACTION PROCESS
# ==============================================================================
# MUST BE SET BEFORE ANY IMPORTS to prevent Apple Silicon crashes
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import numpy as np
import soundfile as sf
import crepe
from keras.models import Model
from tqdm import tqdm

class CREPEFeatureExtractor:
    def __init__(self, model_capacity='full', step_size=10):
        self.step_size = step_size
        self.sr = 16000
        
        self.base_model = crepe.core.build_and_load_model(model_capacity)
        
        embedding_layer = self.base_model.layers[-2].output
        classification_layer = self.base_model.layers[-1].output
        
        self.feature_model = Model(
            inputs=self.base_model.input, 
            outputs=[classification_layer, embedding_layer]
        )

    def extract_features_numpy(self, audio_path: str) -> dict:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
            
        audio = np.pad(audio, 512, mode='constant', constant_values=0)
        
        hop_length = int(self.sr * self.step_size / 1000)
        n_frames = 1 + int((len(audio) - 1024) / hop_length)
        
        # Safe memory allocation
        frames = np.zeros((n_frames, 1024), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            frames[i] = audio[start : start + 1024]
            
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.maximum(np.std(frames, axis=1)[:, np.newaxis], 1e-8)
        
        # ---------------------------------------------------------------------
        # THE FIX: Manual batch loop to completely bypass Keras Multiprocessing
        # ---------------------------------------------------------------------
        activations = []
        embeddings = []
        batch_size = 128
        
        for i in range(0, n_frames, batch_size):
            batch = frames[i : i + batch_size]
            # predict_on_batch avoids the tf.data and multiprocessing backend
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
    processed_dir = Path('./processed/itm_flute')
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Directory {processed_dir} not found.")
        
    stems = [d.name for d in processed_dir.iterdir() if d.is_dir()]
    print(f"Found {len(stems)} stems to process.")
    
    print("Initializing CREPE Feature Extractor (TensorFlow Only)...")
    extractor = CREPEFeatureExtractor(model_capacity='full', step_size=10)
    
    for stem in tqdm(stems, desc="Extracting Audio Features"):
        stem_dir = processed_dir / stem
        audio_path = stem_dir / 'audio_16k.wav'
        npz_path = stem_dir / 'features.npz'
        pt_path = stem_dir / 'features.pt'
        
        # Skip if either the temp file or the final file exists
        if pt_path.exists() or npz_path.exists() or not audio_path.exists():
            continue
            
        try:
            features_np = extractor.extract_features_numpy(str(audio_path))
            np.savez(npz_path, **features_np)
        except Exception as e:
            print(f"\n[ERROR] Failed on {stem}: {e}")

    print("\nExtraction finished. Spawning PyTorch conversion process...")
    # Trigger the isolated conversion pass
    subprocess.run([sys.executable, __file__, '--convert'], check=True)

if __name__ == "__main__":
    main()