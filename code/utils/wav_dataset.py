import os

import torch
from scipy.io.wavfile import read as wav_read

import utils.globals as uglobals

# Raw WAV
class WAVDataset(torch.utils.data.Dataset):
    def __init__(self, midi_path, wav_dir, sr, debug=False):
        self.midi_path = midi_path
        self.sr = sr
        self.debug = debug
        self.midi_dict = torch.load(midi_path)
        self.midi_keys = list(self.midi_dict.keys())

        self.wav_paths = self.get_wav_paths(self.midi_keys, wav_dir)
        print(f'Loaded WAV dataset from {midi_path} with {len(self)} files')

    def get_wav_paths(self, midi_keys, wav_dir):
        wav_paths = []
        for wav_name in sorted(os.listdir(wav_dir)):
            # Match the file name
            midi_name = ('_').join(wav_name.split('_')[:3]) + '.mid'

            # Debug filtering for single instrument
            if self.debug:
                if wav_name.split('_')[-1][:-4] != '41':
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
        wav_name = self.wav_paths[idx].split('wav')[-2][1: -1]
        midi_key = ('_').join(self.wav_paths[idx].split('wav')[1][1: ].split('_')[:-1]) + '.mid'
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
    
def make_wav_loader(midi_path, wav_dir, batch_size, sr, shuffle=True, debug=False):
    dataset = WAVDataset(midi_path, wav_dir, sr, debug=debug)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=wav_collate)#, num_workers=uglobals.NUM_WORKERS, persistent_workers=True)
    return loader