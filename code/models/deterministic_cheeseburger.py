import itertools
import types
import copy

import torch
import lightning
import transformers

from models.pitch_lm import Pitch_LM
from models.wav_lm import WAV_LM
import models.modeling_utils.gpt_partial_forward_patch as gpt_patch

class Adaptor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.config = self.get_config()

        self.gpt = transformers.AutoModel.from_config(self.config)
        self.linear_in = torch.nn.Linear(in_dim, self.config.n_embd)
        self.linear_out = torch.nn.Linear(self.config.n_embd, out_dim)
        
    def get_config(self):
        # Modify the config of a distill-gpt2 model
        config = transformers.AutoConfig.from_pretrained('distilgpt2')
        config.n_embd = 256
        config.n_head = 4
        config.n_layer = 3        
        return config
    
    def forward(self, x):
        x = self.linear_in(x)
        x = self.gpt(inputs_embeds=x).last_hidden_state
        x = self.linear_out(x)
        return x

class Deterministic_Cheeseburger(lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        
        # Modeling
        print(f'Loading WAV LM from checkpoint: {args["det_cheese_wav_lm_checkpoint"]}')
        print(f'... with Spectrogram AE from checkpoint: {args["det_cheese_spectrogram_ae_checkpoint"]}')
        self.wav_lm = WAV_LM(args['det_cheese_spectrogram_ae_checkpoint'])
        self.wav_lm.init_transforms()
        self.wav_lm.load_state_dict(torch.load(args["det_cheese_wav_lm_checkpoint"])['model_state_dict'])

        print(f'Loading Pitch LM from checkpoint: {args["det_cheese_pitch_lm_checkpoint"]}')
        self.pitch_lm = Pitch_LM()
        self.pitch_lm.load_state_dict(torch.load(args["det_cheese_pitch_lm_checkpoint"])['model_state_dict'])

        self.wav_lm.eval()
        self.pitch_lm.eval()

        # Adaptors
        self.adaptor_in = Adaptor(self.wav_lm.config.n_embd, self.pitch_lm.config.n_embd)
        self.adaptor_out = Adaptor(self.pitch_lm.config.vocab_size, self.wav_lm.config.n_embd)

        # Patch the wav lm
        self.wav_lm.gpt.det_cheese_insertion_layer = args['det_cheese_insertion_layer']
        self.wav_lm.gpt.forward_before = types.MethodType(gpt_patch.patched_forward_before, self.wav_lm.gpt)
        self.wav_lm.gpt.forward_after = types.MethodType(gpt_patch.patched_forward_after, self.wav_lm.gpt)
    
    def configure_optimizers(self):
        params = [self.adaptor_in.parameters(), self.adaptor_out.parameters()]
        params_to_update = itertools.chain(*params)
        optimizer = torch.optim.Adam(params_to_update, lr=self.args['lr'])
        return optimizer
    
    def prep_batch(self, batch):
        wav = batch['wav'] # [batch_size, seq_len, n_samples]
        notes = batch['notes'] # [batch_size, seq_len]
        batch_size = wav.shape[0]

        # Zero pad the wav
        wav_in = wav[:, :-1, :]
        padding = torch.zeros_like(wav[:, :1, :])
        wav_in = torch.cat((padding, wav_in), dim=1)

        spectrogram_in = self.wav_lm.transform(wav_in)
        spectrogram_in = self.wav_lm.ae.amp_to_db(spectrogram_in)

        spectrogram = self.wav_lm.transform(wav)
        spectrogram = self.wav_lm.ae.amp_to_db(spectrogram)

        batch_size = wav.shape[0]
        return spectrogram_in, spectrogram, notes, batch_size
    
    def forward_and_ce_loss(self, x, notes_truth):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = self.wav_lm.ae.encode(x) # [batch_size, seq_len, emb_size]
        x = self.wav_lm.gpt.forward_before(inputs_embeds=x)

        x = self.adaptor_in(x)

        x = self.pitch_lm.gpt(inputs_embeds=x).last_hidden_state
        notes_pred = self.pitch_lm.lm_head(x)
        notes_pred = notes_pred.reshape(-1, notes_pred.shape[-1])
        ce_loss = torch.nn.functional.cross_entropy(notes_pred, notes_truth)
        notes_pred_ = notes_pred.argmax(dim=-1)
        notes_pred = notes_pred.reshape(batch_size, seq_len, -1)

        x = self.adaptor_out(notes_pred)
        
        x = self.wav_lm.gpt.forward_after(overwrite_hidden_states=x, inputs_embeds=x).last_hidden_state
        x = self.wav_lm.ae.decode(x, batch_size, seq_len)
        return x, ce_loss, notes_pred_

    def training_step(self, batch, batch_idx):
        spectrogram_in, spectrogram_truth, notes_truth, batch_size = self.prep_batch(batch)
        notes_truth = notes_truth.reshape(-1)
        spectrogram_pred, notes_ce_loss, notes_pred = self.forward_and_ce_loss(spectrogram_in, notes_truth)

        spectrogram_mse_loss = torch.nn.functional.mse_loss(spectrogram_pred, spectrogram_truth)
        
        loss = spectrogram_mse_loss + notes_ce_loss

        self.log('train/loss', loss, batch_size=batch_size) # Automatically averaged
        self.log('train/spectrogram_mse_loss', spectrogram_mse_loss, batch_size=batch_size)
        self.log('train/notes_ce_loss', notes_ce_loss, batch_size=batch_size)
        return loss
    
    def eval_step(self, name, batch, batch_idx):
        spectrogram_in, spectrogram_truth, notes_truth, batch_size = self.prep_batch(batch)
        notes_truth = notes_truth.reshape(-1)
        spectrogram_pred, notes_ce_loss, notes_pred = self.forward_and_ce_loss(spectrogram_in, notes_truth)

        spectrogram_mse_loss = torch.nn.functional.mse_loss(spectrogram_pred, spectrogram_truth)
        
        loss = (2 - self.args['det_cheese_ce_weight']) * spectrogram_mse_loss + self.args['det_cheese_ce_weight'] * notes_ce_loss

        notes_accuracy = (notes_pred == notes_truth).float().mean()

        self.log(f'{name}/loss', loss, batch_size=batch_size) # Automatically averaged
        self.log(f'{name}/spectrogram_mse_loss', spectrogram_mse_loss, batch_size=batch_size)
        self.log(f'{name}/notes_ce_loss', notes_ce_loss, batch_size=batch_size)
        self.log(f'{name}/notes_accuracy', notes_accuracy, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.eval_step('val', batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.eval_step('test', batch, batch_idx)
        return loss

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError
