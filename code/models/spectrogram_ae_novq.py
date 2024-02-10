import datetime
from copy import deepcopy

from scipy.io.wavfile import write as wav_write
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
import torchaudio
import transformers

class Spectrogram_AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = self.get_gpt_config()

        # Each step is an eighth note
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.config.n_embd, kernel_size=(128, 16)),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(self.config.n_embd, self.config.n_embd, kernel_size=(1, 1))
        )
        # Decoder
        self.decoder = torch.nn.Sequential(
            # torch.nn.ConvTranspose2d(self.config.n_embd, self.config.n_embd, kernel_size=(1, 1)),
            # torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(self.config.n_embd, 1, kernel_size=(128, 16))
        )
    
        n_params = sum(p.numel() for p in self.parameters())
        print(f'Initialized spectrogram AE. Number of parameters: {n_params}')

    def get_gpt_config(self):
        # Modify the config of a distill-gpt2 model
        config = transformers.AutoConfig.from_pretrained('distilgpt2')
        return config
    
    def amp_to_db(self, x):
        x = x + 1e-6
        return 10 * torch.log10(x)

    def db_to_amp(self, x):
        return 10 ** (x/10)

    def encode(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(batch_size*seq_len, 128, 16)
        x = x.unsqueeze(1)
        
        x = self.encoder(x)
        x = x.squeeze()
        x = x.reshape(batch_size, seq_len, -1)
        # [batch_size, seq_len, emb_size]
        return x

    def decode(self, x, batch_size, seq_len):
        # [batch_size, seq_len, emb_size]
        x = x.reshape(batch_size*seq_len, -1, 1, 1)
        
        x = self.decoder(x)
        x = x.squeeze()
        x = x.reshape(batch_size, seq_len, 128, 16)
        return x
    
    def forward(self, x):
        # [batch_size, seq_len, 128, 16]
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = self.encode(x)
        x = self.decode(x, batch_size, seq_len)
        return x
    
    def init_crit_and_optim(self):
        # Also setup the transformations
        self.n_fft = 1024
        n_stft = int((self.n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(16000, n_fft=self.n_fft)
        self.invers_transform = torchaudio.transforms.InverseMelScale(sample_rate=16000, n_stft=n_stft)
        self.grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=self.n_fft)
        

    def get_loss(self, pred, y):
        loss = self.criterion(pred, y)
        return loss

    def train_step(self, batch):
        self.train()
        self.optimizer.zero_grad()
        batch = batch.to(self.args.device)
        
        # Transform to spectrogram
        batch = self.transform(batch)

        # AE input/outputs
        x = batch
        y = deepcopy(batch)

        # The model/loss work in the db domain
        x = self.amp_to_db(x)
        y = self.amp_to_db(y)

        # Forward pass
        pred = self(x) # (batch_size, seq_len, 128, 16)

        # Loss
        loss = self.get_loss(pred, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, batch):
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.args.device)
            
            # Transform to spectrogram
            batch = self.transform(batch)

            # AE input/outputs
            x = batch
            y = deepcopy(batch)

            # The model/loss work in the db domain
            x = self.amp_to_db(x)
            y = self.amp_to_db(y)

            # Forward pass
            pred = self(x) # (batch_size, seq_len, 128, 16)

            # Loss
            loss = self.get_loss(pred, y)

            return loss.item()
        
    def spectrogram_to_wav(self, spectrogram):
        out = self.invers_transform(spectrogram)
        out = self.grifflim_transform(out).cpu().numpy()
        return out
        
    def test_recons(self, loader, name, save_folder, n_batch=3):
        for idx, batch in enumerate(loader):
            print(idx)
            if idx >= n_batch:
                break
            batch = self.transform(batch.to(self.args.device))
            ground_truth = deepcopy(batch)
            with torch.no_grad():
                recons = self.forward(batch) # (batch_size, seq_len, 128, 16)

            ground_truth_wav = self.spectrogram_to_wav(ground_truth) # (batch_size, seq_len, wav_len_per_step)
            # Piece together the time steps
            ground_truth_wav = ground_truth_wav.reshape(ground_truth_wav.shape[0], -1)
            recons_wav = self.spectrogram_to_wav(recons)
            recons_wav = recons_wav.reshape(recons_wav.shape[0], -1)

            # Save as wav
            for i in range(ground_truth.shape[0]):
                wav_write(f'{save_folder}/{name}_{idx}_{i}_gt.wav', self.args.uglobals.SR, ground_truth_wav[i])
                wav_write(f'{save_folder}/{name}_{idx}_{i}_decoded.wav', self.args.uglobals.SR, recons_wav[i])
        return