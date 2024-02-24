import datetime
from copy import deepcopy

from scipy.io.wavfile import write as wav_write
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
import torchaudio
import transformers

from models.spectrogram_ae_novq import Spectrogram_AE

class WAV_LM(torch.nn.Module):
    def __init__(self, spectrogram_ae_checkpoint):
        super().__init__()
        self.config = self.get_config()

        self.gpt = transformers.AutoModel.from_config(self.config)

        # Load the pretriained spectrogram AE
        self.ae = Spectrogram_AE()
        self.ae.init_crit_and_optim()
        print(f'loading spectrogram AE checkpoint: {spectrogram_ae_checkpoint}')
        self.ae.load_state_dict(torch.load(spectrogram_ae_checkpoint)['model_state_dict'])
    
        n_params = sum(p.numel() for p in self.parameters())
        print(f'Initialized WAV LM. Number of parameters: {n_params}')
        
    def get_config(self):
        # Modify the config of a distill-gpt2 model
        config = transformers.AutoConfig.from_pretrained('distilgpt2')
        return config
    
    def forward(self, x):
        # [batch_size, seq_len, 128, 16]
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = self.ae.encode(x) # [batch_size, seq_len, emb_size]
        x = self.gpt(inputs_embeds=x).last_hidden_state
        x = self.ae.decode(x, batch_size, seq_len) # [batch_size, seq_len, 128, 16]
        return x
    
    def init_transforms(self):
        self.n_fft = 1024
        n_stft = int((self.n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(16000, n_fft=self.n_fft)
        self.invers_transform = torchaudio.transforms.InverseMelScale(sample_rate=16000, n_stft=n_stft)
        self.grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=self.n_fft)

    def train_step(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        # LM input/outputs
        x = batch[:, :-1, :]
        y = batch[:, 1:, :]

        # Transform to spectrogram
        x = self.transform(x)
        y = self.transform(y)

        # The model/loss work in the db domain
        x = self.ae.amp_to_db(x)
        y = self.ae.amp_to_db(y)

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
            
            # LM input/outputs
            x = batch[:, :-1, :]
            y = batch[:, 1:, :]

            # Transform to spectrogram
            x = self.transform(x)
            y = self.transform(y)

            # The model/loss work in the db domain
            x = self.ae.amp_to_db(x)
            y = self.ae.amp_to_db(y)

            # Forward pass
            pred = self(x) # (batch_size, seq_len, 128, 16)

            # Loss
            loss = self.get_loss(pred, y)

            return loss.item()
        
    def spectrogram_to_wav(self, spectrogram):
        out = self.invers_transform(spectrogram)
        out = self.grifflim_transform(out).cpu().numpy()
        return out
        
    def test_decode(self, loader, name, save_folder, context_len, n_batch=4):
        for idx, batch in enumerate(loader):
            print(idx)
            if idx >= n_batch:
                break
            ground_truth, decoded = self.batch_decode(batch, context_len) # (batch_size, seq_len, 128, 16)

            ground_truth_wav = self.spectrogram_to_wav(ground_truth) # (batch_size, seq_len, wav_len_per_step)
            # Piece together the time steps
            ground_truth_wav = ground_truth_wav.reshape(ground_truth_wav.shape[0], -1)

            decoded_wav = self.spectrogram_to_wav(decoded)
            decoded_wav = decoded_wav.reshape(decoded_wav.shape[0], -1)

            # Save as wav
            for i in range(ground_truth.shape[0]):
                wav_write(f'{save_folder}/{name}_{idx}_{i}_gt.wav', 16000, ground_truth_wav[i])
                wav_write(f'{save_folder}/{name}_{idx}_{i}_decoded.wav', 16000, decoded_wav[i])
        return
    
    def batch_decode(self, batch, context_len):
        context_len = context_len + 1 # Account for BoS
        self.eval()
        with torch.no_grad():

            # Ground truth + transformation/reconstruction oracle
            batch_ground_truth = deepcopy(batch)
            batch_ground_truth = self.transform(batch_ground_truth)
            batch_ground_truth = self.ae.amp_to_db(batch_ground_truth)
            batch_ground_truth = self.ae.db_to_amp(batch_ground_truth)

            batch = self.transform(batch)
            batch = self.ae.amp_to_db(batch)
            x = batch[:, :context_len, :]

            while x.shape[1] < 9:
                pred = self(x)
                pred = pred[:, -1, :]
                x = torch.cat([x, pred.unsqueeze(1)], dim=1)
                
        x = self.ae.db_to_amp(x)
        return batch_ground_truth, x
        
    # Probing: Getting intermediate hidden states
    def forward_get_hidden_states(self, x):
        self.eval()
        with torch.no_grad():
            x = self.transform(x)
            x = self.ae.amp_to_db(x)
            x = self.ae.encode(x)
            x = self.gpt(inputs_embeds=x, output_hidden_states=True).hidden_states
        return x

    def precompute_hidden_states(self, data_utils, uglobals):
        # Make datasets
        train_dataset = data_utils.WAVDataset(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, probing=True)
        dev_dataset = data_utils.WAVDataset(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_WAV_DIR, probing=True)
        
        # Get hidden states
        self.precompute_hidden_state_from_dataset(train_dataset, f'{uglobals.TOY_16K_HIDDEN_STATES_DIR}/train')
        self.precompute_hidden_state_from_dataset(dev_dataset, f'{uglobals.TOY_16K_HIDDEN_STATES_DIR}/dev')
        return
    
    def precompute_hidden_state_from_dataset(self, dataset, save_dir):
        for (x, notes, name) in tqdm(dataset):
            hidden_states = self.forward_get_hidden_states(x.unsqueeze(0)) # tuple of length n_layers + 1 (embedding, with pos emb)
            out = {
                'notes': notes,
                'hidden_states': hidden_states
            }
            torch.save(out, f'{save_dir}/{name}.pt')