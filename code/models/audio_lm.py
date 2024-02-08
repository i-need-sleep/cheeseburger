from pathlib import Path
import itertools
from typing import Any

import torch
import lightning
import transformers

from models.spectogram_rvqvae import Spectorgram_RVQVAE

class AudioLM(lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        
        # Modeling
        print(f'loading spectrogram AE checkpoint: {args["rvqvae_checkpoint"]}\n')
        self.rvqvae = Spectorgram_RVQVAE.load_from_checkpoint(args['rvqvae_checkpoint'])
        self.rvqvae.freeze()
        self.gpt_config = self.get_gpt_config()
        self.gpt = transformers.AutoModel.from_config(self.gpt_config)
        self.bos_emb = torch.nn.Embedding(1, self.gpt_config.n_embd)
        self.cls = torch.nn.Linear(self.gpt_config.n_embd, self.rvqvae.codebook_size * self.rvqvae.n_quantizers)

    def get_gpt_config(self):
        # Modify the config of a distill-gpt2 model
        config = transformers.AutoConfig.from_pretrained(self.args['lm_config'])
        return config
    
    def configure_optimizers(self):
        params_to_update = itertools.chain(self.gpt.parameters(), self.bos_emb.parameters(), self.cls.parameters()) # The RVQVAE is frozen
        optimizer = torch.optim.Adam(params_to_update, lr=self.args['lr'])
        return optimizer
    
    def batch_to_tokens(self, batch):
        x = batch['wav'] # [batch_size, seq_len, n_samples]
        batch_size = x.shape[0]
        spectrogram = self.rvqvae.preprocess(x)
        quantized, indices, _ = self.rvqvae.encode_and_quantize(spectrogram)
        quantized = quantized.reshape(batch_size, -1, quantized.shape[-1])
        indices = indices.reshape(batch_size, -1, indices.shape[-1]) # [batch_size, seq_len, n_quantizers]
        return quantized, indices, spectrogram
    
    def get_bos_emb(self, batch_size):
        return self.bos_emb(torch.zeros(batch_size, 1, dtype=torch.long, device=self.device))
    
    def prep_input_outputs(self, batch):
        quantized, tokens, spectrogram = self.batch_to_tokens(batch)
        # quantized: [batch_size, seq_len, emb_size]
        # tokens: [batch_size, seq_len, n_quantizers]
        embs = self.get_bos_emb(quantized.shape[0])
        embs = torch.cat([embs, quantized], dim=1)
        
        # Shift the input tokens
        embs = embs[:, :-1] # [batch_size, seq_len, emb_size]
        return embs, tokens, spectrogram

    def forward(self, x):
        x = self.gpt(inputs_embeds=x).last_hidden_state # [batch_size, seq_len, emb_size]
        x = self.cls(x) # [batch_size, seq_len, codebook_size * n_quantizers]
        x = x.reshape(x.shape[0], x.shape[1], self.rvqvae.n_quantizers, self.rvqvae.codebook_size)
        return x

    def training_step(self, batch, batch_idx):
        batch_size = batch['wav'].shape[0]
        embs, gt_tokens, _ = self.prep_input_outputs(batch)
        preds = self(embs) # [batch_size, seq_len, n_quantizers, codebook_size]
        gt_tokens = gt_tokens.reshape(-1)
        preds = preds.reshape(-1, preds.shape[-1])
        loss = torch.nn.functional.cross_entropy(preds, gt_tokens)
        
        self.log('train/loss', loss, batch_size=batch_size)
        return loss
    
    def eval_step(self, name, batch, batch_idx):
        batch_size = batch['wav'].shape[0]
        embs, gt_tokens, spectrogram = self.prep_input_outputs(batch)
        preds = self(embs) # [batch_size, seq_len, n_quantizers, codebook_size]
        gt_tokens = gt_tokens.reshape(-1)
        preds = preds.reshape(-1, preds.shape[-1])
        loss = torch.nn.functional.cross_entropy(preds, gt_tokens)
        accuracy = (preds.argmax(dim=-1) == gt_tokens).float().mean()
        
        self.log(f'{name}/loss', loss, batch_size=batch_size) # Automatically averaged
        self.log(f'{name}/accuracy', accuracy, batch_size=batch_size)

        # Reconstruction
        preds = preds.argmax(dim=-1).reshape(-1, self.rvqvae.n_quantizers)
        preds = self.rvqvae.decode_from_indices(preds, batch_size)
        spectrogram = spectrogram.reshape(batch_size, -1, spectrogram.shape[-2], spectrogram.shape[-1])
        recons_loss = torch.nn.functional.mse_loss(preds, spectrogram)
        self.log(f'{name}/recons_mse', recons_loss, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.eval_step('val', batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.eval_step('test', batch, batch_idx)
        return loss

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError
