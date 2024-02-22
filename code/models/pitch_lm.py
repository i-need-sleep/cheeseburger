from pathlib import Path
import itertools
from copy import deepcopy

import torch
import torchaudio
import lightning
from lightning.pytorch.utilities import grad_norm

import transformers

class PitchLM(lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        
        # Modeling: Transformer LM
        self.gpt_config = self.get_gpt_config()
        self.gpt = transformers.AutoModel.from_config(self.gpt_config)
        self.lm_head = torch.nn.Linear(self.gpt_config.n_embd, self.gpt_config.vocab_size)

    def get_gpt_config(self):
        # Modify the config of a distill-gpt2 model
        config = transformers.AutoConfig.from_pretrained(self.args['pitch_lm_config'])
        config.vocab_size = 128
        config.bos_token_id = 0
        config.eos_token_id = 1
        return config
    
    # Optimization
    def configure_optimizers(self):
        params = [self.gpt.parameters(), self.lm_head.parameters()]
        self.params_to_update = itertools.chain(*params)
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.args['lr'])

        # LR scheduler
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.args['lr_scheduler_start_factor'], end_factor=1, total_iters=self.args['lr_scheduler_warmup_epochs'])
        scheduler_anneal = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=self.args['lr_scheduler_end_factor'], total_iters=self.args['lr_scheduler_anneal_epochs'])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler_warmup, scheduler_anneal], milestones=[self.args['lr_scheduler_warmup_epochs']])
        return [optimizer], [scheduler]
    
    def on_before_optimizer_step(self, optimizer):
        # Track the gradient norms
        grad_norms = grad_norm(self, norm_type=2)['grad_2.0_norm_total']
        self.log('train/grad_norms', grad_norms, batch_size=1)

    # Data Processing
    def preprocess(self, notes):
        # Zero pad left
        notes = torch.nn.functional.pad(notes, (1, 0), value=0)
        notes_input = notes[:, :-1]
        notes_target = notes[:, 1:]
        return notes_input, notes_target

    # Forward Pass
    def forward(self, x):
        x = self.gpt(x).last_hidden_state
        x = self.lm_head(x)
        return x

    # Training
    def training_step(self, batch, batch_idx):
        notes = batch['notes']
        batch_size = notes.shape[0]
        notes_input, notes_target = self.preprocess(notes)
        logits = self(notes_input)

        loss = torch.nn.CrossEntropyLoss()(logits.reshape(-1, logits.shape[-1]), notes_target.reshape(-1))
        self.log('train/loss', loss, batch_size=batch_size)
        return loss
    
    def eval_step(self, name, batch, batch_idx):
        notes = batch['notes']
        batch_size = notes.shape[0]
        notes_input, notes_target = self.preprocess(notes)
        logits = self(notes_input)

        loss = torch.nn.CrossEntropyLoss()(logits.reshape(-1, logits.shape[-1]), notes_target.reshape(-1))
        accuracy = (logits.argmax(-1) == notes_target).float().mean()
        self.log(f'{name}/loss', loss, batch_size=batch_size) # Automatically averaged
        self.log(f'{name}/accuracy', accuracy, batch_size=batch_size)
        self.log(f'{name}/monitor', accuracy, batch_size=batch_size) # Keep the best checkpoint based on this metric
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.eval_step('val', batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.eval_step('test', batch, batch_idx)
        return loss

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError
