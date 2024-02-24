from pathlib import Path
import itertools

import torch
import lightning
from lightning.pytorch.utilities import grad_norm
from scipy.io.wavfile import write as wav_write

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
        
        # Projecting the quantized code to the LM inputs if they are of different sizes
        self.input_proj = None
        if self.rvqvae.codebook_dim != self.gpt_config.n_embd:
            self.input_proj = torch.nn.Linear(self.rvqvae.codebook_dim, self.gpt_config.n_embd)

        # Initialize the output folder
        self.output_folder = f'{args["uglobals"]["OUTPUTS_DIR"]}/{args["task"]}/{args["name"]}'
        if args['mode'] == 'predict_dev':
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def get_gpt_config(self):
        # Modify the config of a distill-gpt2 model
        config = transformers.AutoConfig.from_pretrained(self.args['lm_config'])
        return config
    
    def configure_optimizers(self):
        params = [self.gpt.parameters(), self.bos_emb.parameters(), self.cls.parameters()] # The RVQVAE is frozen
        if self.input_proj is not None:
            params.append(self.input_proj.parameters()) 
        self.params_to_update = itertools.chain(*params)
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.args['lr'])

        # LR scheduler
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.args['lr_scheduler_start_factor'], end_factor=1, total_iters=self.args['lr_scheduler_warmup_epochs'])
        scheduler_anneal = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=self.args['lr_scheduler_end_factor'], total_iters=self.args['lr_scheduler_anneal_epochs'])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler_warmup, scheduler_anneal], milestones=[self.args['lr_scheduler_warmup_epochs']])
        return [optimizer], [scheduler]
    
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
        if self.input_proj is not None:
            # Project the quantized code to the LM inputs if they are of different sizes
            quantized = self.input_proj(quantized)

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
    
    def infer(self, tokens, context_len):
        # x: [batch_size, seq_len, emb_size]
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]
        
        tokens = tokens[:, :context_len, :] # [batch_size, known_context_len, n_quantizers]
        while tokens.shape[1] < seq_len:
            tokens = tokens.reshape(-1, tokens.shape[-1])
            embs = self.rvqvae.rvq.get_output_from_indices(tokens)
            embs = embs.reshape(batch_size, -1, embs.shape[-1]) # [batch_size, seq_len, emb_size]
            if self.input_proj is not None:
                embs = self.input_proj(embs)
            embs = torch.cat([self.get_bos_emb(batch_size), embs], dim=1)
            embs = self.gpt(inputs_embeds=embs).last_hidden_state
            embs = self.cls(embs) # [batch_size, seq_len, codebook_size * n_quantizers]
            embs = embs.reshape(embs.shape[0], embs.shape[1], self.rvqvae.n_quantizers, self.rvqvae.codebook_size)[:, -1:, :, :] # [batch_size, n_quantizers, codebook_size]
            pred = embs.argmax(dim=-1) # [batch_size, 1, n_quantizers]
            tokens = torch.cat([tokens.reshape(batch_size, -1, self.rvqvae.n_quantizers), pred], dim=1)
        return tokens

    def training_step(self, batch, batch_idx):
        batch_size = batch['wav'].shape[0]
        embs, gt_tokens, _ = self.prep_input_outputs(batch)
        preds = self(embs) # [batch_size, seq_len, n_quantizers, codebook_size]
        gt_tokens = gt_tokens.reshape(-1)
        preds = preds.reshape(-1, preds.shape[-1])
        loss = torch.nn.functional.cross_entropy(preds, gt_tokens)
        
        self.log('train/loss', loss, batch_size=batch_size)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size)
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        # Track the gradient norms
        grad_norms = grad_norm(self, norm_type=2)['grad_2.0_norm_total']
        self.log('train/grad_norms', grad_norms, batch_size=1)
    
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

    @torch.enable_grad() 
    def predict_step(self, batch, batch_idx):
        context_len = 4
        sr = 16000

        batch_size = batch['wav'].shape[0]
        embs, gt_tokens, spectrogram = self.prep_input_outputs(batch)
        pred = self.infer(gt_tokens, context_len=context_len)

        # Reconstruction
        pred = pred.reshape(-1, self.rvqvae.n_quantizers)
        pred = self.rvqvae.decode_from_indices(pred, batch_size).detach()
        spectrogram = spectrogram.reshape(batch_size, -1, spectrogram.shape[-2], spectrogram.shape[-1])

        gt_tokens = gt_tokens.reshape(-1, self.rvqvae.n_quantizers)
        vae_spectrogram = self.rvqvae.decode_from_indices(gt_tokens, batch_size).detach()
        vae_spectrogram = vae_spectrogram.reshape(batch_size, -1, vae_spectrogram.shape[-2], vae_spectrogram.shape[-1])
        
        wav_original = batch['wav'].reshape(batch_size, -1)
        wav_processed = self.rvqvae.postprocess(spectrogram, batch_size).reshape(batch_size, -1)
        wav_vae = self.rvqvae.postprocess(vae_spectrogram, batch_size).reshape(batch_size, -1)
        wav_pred = self.rvqvae.postprocess(pred, batch_size).reshape(batch_size, -1)

        names = batch['names']
        for i, name in enumerate(names):
            wav_write(f'{self.output_folder}/{name}_original.wav', sr, wav_original[i].cpu().numpy())
            wav_write(f'{self.output_folder}/{name}_processed.wav',  sr, wav_processed[i].cpu().numpy())
            wav_write(f'{self.output_folder}/{name}_vae_recons.wav',  sr, wav_vae[i].cpu().numpy())
            wav_write(f'{self.output_folder}/{name}_pred.wav',  sr, wav_pred[i].cpu().numpy())
        return
