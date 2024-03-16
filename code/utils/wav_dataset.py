import os
import itertools

import torch
from scipy.io.wavfile import read as wav_read

import utils.globals as uglobals

# Raw WAV
class WAVDataset(torch.utils.data.Dataset):
    def __init__(self, midi_path, wav_dir, sr, single_instrument=False):
        self.midi_path = midi_path
        self.sr = sr
        self.single_instrument = single_instrument
        self.midi_dict = torch.load(midi_path)
        self.midi_keys = list(self.midi_dict.keys())

        self.wav_paths = self.get_wav_paths(self.midi_keys, wav_dir)
        print(f'Loaded WAV dataset from {midi_path} with {len(self)} files')

    def get_wav_paths(self, midi_keys, wav_dir):
        wav_paths = []
        for wav_name in sorted(os.listdir(wav_dir)):
            # Match the file name
            midi_name = ('_').join(wav_name.split('_')[:3]) + '.mid'

            # Filtering for single instrument
            if self.single_instrument:
                if wav_name.split('_')[-1][:-4] != '0':
                    continue

            if midi_name in midi_keys:
                file_path = os.path.join(wav_dir, wav_name)
                wav_paths.append(file_path)
        return wav_paths
    
    def __len__(self):
        return len(self.wav_paths)
    
    def __getitem__(self, idx):
        
        wav = torch.tensor(wav_read(self.wav_paths[idx])[1])
        # Slice into chunks of length SR // 2 (eighth note)
        wav = wav.reshape(-1, self.sr//2).float()
        # Resolve the midi key
        wav_name = os.path.split(self.wav_paths[idx])[1].replace('.mid', '')
        midi_key = '_'.join(wav_name.split('_')[:-1]) + '.mid'
        midi = self.midi_dict[midi_key]
        
        return wav, midi, wav_name
    
def wav_collate(batch):
    wav = [b[0] for b in batch]
    wav = torch.stack(wav)
    notes = torch.stack([torch.tensor(b[1]) for b in batch])
    names = [b[2] for b in batch]
    return {
        'wav': wav,
        'notes': notes,
        'names': names
    }
    
def make_wav_loader(midi_path, wav_dir, batch_size, sr, shuffle=True, single_instrument=False, single_worker=False):
    dataset = WAVDataset(midi_path, wav_dir, sr, single_instrument=single_instrument)
    if single_worker:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=wav_collate)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=wav_collate, num_workers=uglobals.NUM_WORKERS, persistent_workers=True)
    return loader

# Raw WAV with augmentation: Three sets
# Set 1: Randomly drawn timbre doner
# Set 2: Randomly drawn pitch doner
# Set 3: Ground truth for the recombined pitch-timbre pair
class AugmentedWavDataset(torch.utils.data.Dataset):
    def __init__(self, midi_path, wav_dir, sr):
        self.midi_path = midi_path
        self.sr = sr
        self.midi_dict = torch.load(midi_path)
        self.midi_keys = list(self.midi_dict.keys())

        self.wav_paths = self.get_wav_paths(self.midi_keys, wav_dir)
        print(f'Loaded WAV dataset from {midi_path} with {len(self)} files')

    def get_wav_paths(self, midi_keys, wav_dir):
        wav_paths = []
        for wav_name in sorted(os.listdir(wav_dir)):
            # Match the file name
            midi_name = ('_').join(wav_name.split('_')[:3]) + '.mid'

            if midi_name in midi_keys:
                file_path = os.path.join(wav_dir, wav_name)
                wav_paths.append(file_path)
        return wav_paths
    
    def __len__(self):
        return len(self.wav_paths) ** 2 # All combinations
    
    def wav_path_to_tensor_and_notes(self, path):
        wav = torch.tensor(wav_read(path)[1])
        # Slice into chunks of length SR // 2 (eighth note)
        wav = wav.reshape(-1, self.sr//2).float()

        # Resolve the midi key
        wav_name = path.split('wav_flat_velo')[-1][1: -4]
        midi_key = ('_').join(path.split('wav_flat_velo')[1][1: ].split('_')[:-1]) + '.mid'
        midi = torch.tensor(self.midi_dict[midi_key])
        return wav, midi, wav_name
    
    def __getitem__(self, idx):
        timbre_idx = idx // len(self.wav_paths)
        pitch_idx = idx % len(self.wav_paths)

        timbre_wav, timbre_midi, timbre_name = self.wav_path_to_tensor_and_notes(self.wav_paths[timbre_idx])
        pitch_wav, pitch_midi, pitch_name = self.wav_path_to_tensor_and_notes(self.wav_paths[pitch_idx])

        # Resolve the path for the recombined pair
        combined_path = '_'.join(pitch_name.split('_')[: -1]) + '_' + timbre_name.split('_')[-1] + '.wav'
        combined_wav, combined_midi, combined_name = self.wav_path_to_tensor_and_notes(os.path.join(uglobals.TOY_16K_FLAT_VELO_WAV_DIR, combined_path))

        # Stack everything
        wav = torch.stack([timbre_wav, pitch_wav, combined_wav]) # [3, 8, 8000]
        notes = torch.stack([timbre_midi, pitch_midi, combined_midi]) # [3, 8]
        names = [timbre_name, pitch_name, combined_name]
        return wav, notes, names
    
def augmented_wav_collate(batch):
    wav = torch.cat([b[0] for b in batch])
    notes = torch.cat([b[1] for b in batch])
    names = list(itertools.chain(*[b[2] for b in batch]))
    return {
        'wav': wav,
        'notes': notes,
        'names': names
    }
    
def make_augmented_wav_loader(midi_path, wav_dir, batch_size, sr, shuffle=True, single_worker=False):
    dataset = AugmentedWavDataset(midi_path, wav_dir, sr)
    if single_worker:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=augmented_wav_collate)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=augmented_wav_collate, num_workers=uglobals.NUM_WORKERS, persistent_workers=True)
    return loader
