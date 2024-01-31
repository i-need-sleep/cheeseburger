import os
import copy
import random
import shutil

import torch, torchaudio
import numpy as np
from scipy.io.wavfile import write as wav_write
from scipy.io.wavfile import read as wav_read
import pretty_midi
from tqdm import tqdm

import utils.globals as uglobals
from utils.wav_dataset import make_wav_loader

def augment_midi(in_folder, out_folder, transpose_down, transpose_up):
    # Transpose down/up in semitones (inclusive)
    for file_name in os.listdir(in_folder):
        file_path = os.path.join(in_folder, file_name)
        midi = pretty_midi.PrettyMIDI(file_path)
        notes = midi.instruments[0].notes

        for i in range(transpose_down, transpose_up + 1):
            out_midi = copy.deepcopy(midi)
            out_midi.instruments[0].notes = []

            for note in notes:
                out_midi.instruments[0].notes.append(pretty_midi.Note(
                    start=note.start,
                    end=note.end,
                    pitch=note.pitch + i,
                    velocity=note.velocity
                ))
            out_midi.write(os.path.join(out_folder, f'{file_name[:-4]}_{i}.mid'))
    return

def render_midi(in_folder, out_folder, sr, total_time):
    # Render MIDI files as WAV files
    # Apply a velocity curve
    # Syntehsize for all instruments
    for file_name in tqdm(os.listdir(in_folder)):
        file_path = os.path.join(in_folder, file_name)
        midi = pretty_midi.PrettyMIDI(file_path)
        
        # Velocity
        notes = midi.instruments[0].notes
        notes = apply_velo(notes)
        
        # Instruments
        for instrument_program in range(128):
            wav_path = os.path.join(out_folder, f'{file_name[:-4]}_{instrument_program}.wav')
            render_note_by_program(notes, instrument_program, wav_path, sr=sr, total_time=total_time)
            

def render_note_by_program(notes, program, out_path, sr, total_time):
    midi = pretty_midi.PrettyMIDI(initial_tempo=60)
    instrument = pretty_midi.Instrument(program=program)
    instrument.notes = notes
    midi.instruments.append(instrument)

    wav = midi.fluidsynth(fs=float(sr))

    # Pad
    total_samples = int(sr * total_time)
    if wav.shape[0] < total_samples:
        wav = np.pad(wav, (0, total_samples - wav.shape[0]), 'constant')

    # Cut off the trailing slience
    wav = wav[: total_samples]
    wav_write(out_path, sr, wav)
    return

def pack_and_write_midi(name, file_names, in_folder, out_folder):
    # Pack MIDI into a dict for each split {file_name: notes}
    midi_dict = {}
    for file_name in file_names:
        file_path = os.path.join(in_folder, file_name)
        midi = pretty_midi.PrettyMIDI(file_path)
        notes = midi.instruments[0].notes

        # Convert to pitches
        notes = [note.pitch for note in notes]
        midi_dict[file_name] = notes
        
    # Write to a file
    torch.save(midi_dict, os.path.join(out_folder, f'{name}_midi.pt'))
    return

def make_splits(midi_dir, out_dir, dev_ratio, test_ratio):
    # Split by augmented MIDI
    midi_files = os.listdir(midi_dir)
    random.shuffle(midi_files)

    num_midis = len(midi_files)
    num_dev = int(num_midis * dev_ratio)
    num_test = int(num_midis * test_ratio)

    dev_midi_files = midi_files[:num_dev]
    test_midi_files = midi_files[num_dev:num_dev + num_test]
    train_midi_files = midi_files[num_dev + num_test:]

    # Pack MIDI into a dict for each split {file_name: notes}
    for split_name, split in [['train', train_midi_files], ['dev', dev_midi_files], ['test', test_midi_files]]:
        pack_and_write_midi(split_name, split, midi_dir, out_dir)
    return

def apply_velo(notes, max_velo=127, min_velo=40, noise_size=4, min_diff=45):
    # Apply a linearly ascending or descending velocity curve
    start_velo = random.randint(min_velo, max_velo)
    end_velo = random.randint(min_velo, max_velo)

    # Make sure that the velo range is large enough
    while abs(start_velo - end_velo) < min_diff:
        start_velo = random.randint(min_velo, max_velo)
        end_velo = random.randint(min_velo, max_velo)

    if start_velo < end_velo:
        for i, note in enumerate(notes):
            noise = random.randint(-noise_size, noise_size)
            velo = int(start_velo + (end_velo - start_velo) * note.start / 16) + noise
            velo = max(velo, min_velo)
            velo = min(velo, max_velo)
            notes[i].velocity = velo
    else:
        for i, note in enumerate(notes):
            noise = random.randint(-noise_size, noise_size)
            velo = int(start_velo - (start_velo - end_velo) * note.start / 16) + noise
            velo = max(velo, min_velo)
            velo = min(velo, max_velo)
            notes[i].velocity = velo
    return notes

def get_normalization_factors(train_pt, wav_dir, sr, batch_size):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    transform = torchaudio.transforms.MelSpectrogram(16000, n_fft=uglobals.N_FFT).to(device)

    train_loader = make_wav_loader(train_pt, wav_dir, batch_size, sr, shuffle=True)

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        wav = batch['wav']
        spectorgram = transform(wav.to(device))
        spectorgram = 10 * torch.log10(spectorgram)
        spectorgram.to('cpu')
        
        if batch_idx == 0:
            spectorgrams = spectorgram
        else:
            spectorgrams = torch.cat([spectorgrams, spectorgram], dim=0)
    
    mean = spectorgrams.mean()
    std = spectorgrams.std()
    print(f'Mean: {mean}, std: {std}')

if __name__ == '__main__':
    # Scale runs
    # augment_midi(uglobals.TOY_MIDI_PROTOTYPES_DIR, uglobals.TOY_MIDI_AUGMENTED_DIR, -24, 24)
    # render_midi(uglobals.TOY_16K_MIDI_AUGMENTED_DIR, uglobals.TOY_16K_WAV_DIR, sr=16000, total_time=4)
    # make_splits(uglobals.TOY_MIDI_AUGMENTED_DIR, uglobals.TOY_TRAINING_DIR, 0.1, 0.1)
    get_normalization_factors(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, 16000, 32)