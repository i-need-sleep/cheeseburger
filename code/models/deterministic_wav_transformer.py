from pathlib import Path
import itertools
from copy import deepcopy

import torch
import torchaudio
import lightning
from lightning.pytorch.utilities import grad_norm
from scipy.io.wavfile import write as wav_write

import transformers

class DeterministicWavTransformer(lightning.LightningModule):
    def __init__(self, args, sr):
        super().__init__()
        self.args = args
        self.sr = sr
        self.save_hyperparameters()
        
        # Modeling: Transformer
        self.gpt_config = self.get_gpt_config()
        self.gpt = transformers.AutoModel.from_config(self.gpt_config)
        # We don't really need a BoS step for this deterministic model, but we keep it for compatibility with the pitch model
        self.bos_emb = torch.nn.Embedding(1, self.gpt_config.n_embd)

        # Modeling: Encoder/Decoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.gpt_config.n_embd, kernel_size=(128, 16)),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.gpt_config.n_embd, 1, kernel_size=(128, 16))
        )

        # Transformations
        n_fft = self.args['uglobals']['N_FFT']
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft)
        self.invers_transform = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=n_stft)
        self.grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft)

        # Initialize the output folder
        self.output_folder = f'{args["uglobals"]["OUTPUTS_DIR"]}/{args["task"]}/{args["name"]}'
        if args['mode'] == 'predict_dev':
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def get_gpt_config(self):
        # Modify the config of a distill-gpt2 model
        config = transformers.AutoConfig.from_pretrained(self.args['lm_config'])
        return config
    
    # Optimization
    def configure_optimizers(self):
        params = [self.gpt.parameters(), self.bos_emb.parameters(), self.encoder.parameters(), self.decoder.parameters()]
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
    def preprocess_wav(self, x):
        x = self.transform(x) # Wav to mel spectrogram
        x = 10 * torch.log10(x + 1e-6) # Log scale
        x = (x - self.args['uglobals']['SPECTORGRAM_MEAN']) / self.args['uglobals']['SPECTORGRAM_STD'] # Normalize
        return x

    def postprocess_wav(self, x, batch_size):
        x = (x * self.args['uglobals']['SPECTORGRAM_STD']) + self.args['uglobals']['SPECTORGRAM_MEAN'] # Denormalize
        x = torch.pow(10, x / 10) - 1e-6 # Inverse log scale
        x = self.invers_transform(x) # Mel spectrogram to wav
        x = self.grifflim_transform(x)
        return x

    # Forward Pass
    def get_bos_emb(self, batch_size):
        return self.bos_emb(torch.zeros(batch_size, 1, dtype=torch.long, device=self.device))
    
    def forward(self, x):
        # x: [batch_size, seq_len-1, 128, 16]
        batch_size = x.shape[0]
        seq_len = x.shape[1] + 1

        x = x.reshape(batch_size * (seq_len - 1), 1, x.shape[2], x.shape[3]) # [batch_size * (seq_len-1), 1, 128, 16]
        x = self.encoder(x)
        x = x.reshape(batch_size, seq_len - 1, -1) # [batch_size, seq_len-1, emb_size]

        bos = self.get_bos_emb(batch_size)
        x = torch.cat([bos, x], dim=1) # [batch_size, seq_len, emb_size], pad the BoS token
        x = self.gpt(inputs_embeds=x).last_hidden_state # [batch_size, seq_len, emb_size]

        x = x.reshape(batch_size * seq_len, -1, 1, 1) # [batch_size * seq_len, emb_size, 1, 1]
        x = self.decoder(x)
        x = x.reshape(batch_size, seq_len, x.shape[2], x.shape[3]) # [batch_size, seq_len, 128, 16]
        return x

    def infer(self, x, context_len):
        # x: [batch_size, seq_len-1, 128, 16]
        batch_size = x.shape[0]
        seq_len = x.shape[1] + 1

        x = x[:, :context_len, :, :]

        while x.shape[1] < seq_len:
            emb = x.reshape(-1, 1, x.shape[2], x.shape[3]) # [batch_size * (seq_len-1), 1, 128, 16]
            emb = self.encoder(emb)
            emb = emb.reshape(batch_size, -1, self.gpt_config.n_embd) # [batch_size, seq_len-1, emb_size]

            bos = self.get_bos_emb(batch_size)
            emb = torch.cat([bos, emb], dim=1) # [batch_size, seq_len, emb_size], pad the BoS token

            pred = self.gpt(inputs_embeds=emb).last_hidden_state[:, -1:, :] # [batch_size, 1, emb_size]
            pred = pred.reshape(batch_size, -1, 1, 1)
            pred = self.decoder(pred)
            pred = pred.reshape(batch_size, 1, pred.shape[2], pred.shape[3])
            x = torch.cat([x, pred], dim=1)
        return x
    
    # Training
    def training_step(self, batch, batch_idx):
        wav = batch['wav']
        batch_size = wav.shape[0]
        x = self.preprocess_wav(wav) # spectrogram: [batch_size, seq_len, 128, 16]
        x_true = deepcopy(x)
        x = self(x[:, :-1, :, :]) # x: [batch_size, seq_len-1, 128, 16]
        loss = torch.nn.functional.mse_loss(x, x_true)

        self.log('train/loss', loss, batch_size=batch_size) 
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size)
        return loss
    
    def eval_step(self, name, batch, batch_idx):
        wav = batch['wav']
        batch_size = wav.shape[0]
        x = self.preprocess_wav(wav) # spectrogram: [batch_size, seq_len, 128, 16]
        x_true = deepcopy(x)
        x = self(x[:, :-1, :, :]) # x: [batch_size, seq_len-1, 128, 16]
        loss = torch.nn.functional.mse_loss(x, x_true)

        self.log(f'{name}/loss', loss, batch_size=batch_size) # Automatically averaged
        self.log(f'{name}/monitor', loss, batch_size=batch_size) # Monitor the loss
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

        wav = batch['wav']
        batch_size = wav.shape[0]
        wav_original = deepcopy(wav)
        x = self.preprocess_wav(wav) # spectrogram: [batch_size, seq_len, 128, 16]
        spectrogram_oracle = deepcopy(x)
        spectrogram_pred = self.infer(x[:, :-1, :, :], context_len).detach()

        wav_original = wav_original.reshape(batch_size, -1).cpu().numpy()
        wav_oracle = self.postprocess_wav(spectrogram_oracle, batch_size).reshape(batch_size, -1).cpu().numpy()
        wav_pred = self.postprocess_wav(spectrogram_pred, batch_size).reshape(batch_size, -1).cpu().numpy()

        names = batch['names']
        for i, name in enumerate(names):
            wav_write(f'{self.output_folder}/{name}_original.wav', self.sr, wav_original[i])
            wav_write(f'{self.output_folder}/{name}_oracle.wav',  self.sr, wav_oracle[i])
            wav_write(f'{self.output_folder}/{name}_pred.wav',  self.sr, wav_pred[i])
        return
