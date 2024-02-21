import datetime

from torch.utils.tensorboard import SummaryWriter

import torch
import transformers

class Pitch_LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = self.get_config()

        self.gpt = transformers.AutoModel.from_config(self.config)
        self.lm_head = torch.nn.Linear(self.config.n_embd, self.config.vocab_size, bias=True)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f'Initialized MIDI LM. Number of parameters: {n_params}')
        
    def get_config(self):
        # Modify the config of a distill-gpt2 model
        config = transformers.AutoConfig.from_pretrained('distilgpt2')
        config.vocab_size = 128
        config.bos_token_id = 0
        config.eos_token_id = 1

        config.n_embd = 256
        config.n_head = 4
        config.n_layer = 3        
        return config
    
    def forward(self, x):
        x = self.gpt(x).last_hidden_state
        x = self.lm_head(x)
        return x
    
    def get_loaders_toy(self, args, data_utils, uglobals):
        # Unused, for the scale runs
        train_loader = data_utils.make_MIDI_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', args.batch_size, shuffle=True)
        dev_loader = data_utils.make_MIDI_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', args.batch_size, shuffle=False)
        return train_loader, dev_loader

    def get_loaders(self, args, data_utils, uglobals):
        train_loader = data_utils.make_MIDI_loader(f'{uglobals.NOTTINGHAM_TRAINING_DIR}/train_midi.pt', args.batch_size, shuffle=True)
        dev_loader = data_utils.make_MIDI_loader(f'{uglobals.NOTTINGHAM_TRAINING_DIR}/dev_midi.pt', args.batch_size, shuffle=False)
        return train_loader, dev_loader
    
    def init_crit_and_optim(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def train_step(self, batch):
        self.train()
        self.optimizer.zero_grad()
        batch = batch.to(self.args.device)
        
        # LM input/outputs
        x = batch[:, :-1]
        y = batch[:, 1:]

        # Forward pass
        logits = self(x) # (batch_size, seq_len, vocab_size)

        # Loss
        loss = self.criterion(logits.reshape(-1, self.config.vocab_size), y.reshape(-1))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def eval_step(self, batch):
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.args.device)
            
            # LM input/outputs
            x = batch[:, :-1]
            y = batch[:, 1:]

            # Forward pass
            logits = self(x) # (batch_size, seq_len, vocab_size)

            # Loss
            loss = self.criterion(logits.reshape(-1, self.config.vocab_size), y.reshape(-1))
    
            return loss.item()
        
    def batch_decode(self, batch, total_len=9):
        self.eval()
        batch = batch.to(self.args.device)
        with torch.no_grad():
            while batch.shape[1] < total_len:
                logits = self(batch)
                predicted = torch.argmax(logits, dim=-1)
                batch = torch.cat([batch, predicted[:, -1:]], dim=1)
        return batch
    
    def test_decode_acc(self, loader, context_len=3):
        n_hit = 0
        n_total = 0
        start_idx = context_len + 1 # Account for BoS
        for batch in loader:
            batch_x = batch[:, :start_idx] # The first 3 notes + BoS determines the sequence
            decoded = self.batch_decode(batch_x)

            decoded_y = decoded[:, start_idx:]
            batch_y = batch[:, start_idx:]

            n_hit = n_hit + (decoded_y == batch_y).sum().item()
            n_total = n_total + decoded_y.numel()
        acc = n_hit / n_total
        return acc, n_hit, n_total