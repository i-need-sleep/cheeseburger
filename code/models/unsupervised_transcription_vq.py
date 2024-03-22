from pathlib import Path
import itertools
from scipy.io.wavfile import write as wav_write

import torch
import lightning
import torchaudio, torchvision
from lightning.pytorch.utilities import grad_norm
from vector_quantize_pytorch import VectorQuantize

from models.pitch_lm import PitchLM
from models.modeling_utils.rerank_vq import RerankVQ
from models.modeling_utils.res_module import ResModule

class UnsupervisedTranscriptionVQ(lightning.LightningModule):
    def __init__(self, args, sr):
        super().__init__()
        self.args = args
        self.sr = sr
        self.save_hyperparameters()

        # Transformations
        n_fft = self.args['uglobals']['N_FFT']
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft)
        self.invers_transform = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=n_stft)
        self.grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft)

        # Modeling: Encoder/Decoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(2, 2)),
            torch.nn.ReLU(),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.Conv2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),ResModule(
                torch.nn.Sequential(
                    torch.nn.Conv2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            torch.nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(2, 2)),
            torch.nn.ReLU(),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            torch.nn.Conv2d(16, 768, kernel_size=(128, 16)),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(768, 16, kernel_size=(128, 16)),
            torch.nn.ReLU(),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=(5, 5), padding=(2, 2)),
            torch.nn.ReLU(),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            torch.nn.ConvTranspose2d(8, 1, kernel_size=(5, 5), padding=(2, 2)),
        )

        # Modeling: Pretrained, pitch model
        print(f'Loading Pitch LM from checkpoint: {args["det_cheese_pitch_lm_checkpoint"]}')
        self.pitch_lm = PitchLM.load_from_checkpoint(args["det_cheese_pitch_lm_checkpoint"])
        self.pitch_lm.freeze()

        # Modeling: Custom VQ layer
        self.vq = RerankVQ(dim=768, codebook_size=int(self.pitch_lm.gpt_config.vocab_size * self.args['unsupervised_transcription_vq_codebook_size_factor']))
        # self.vq = VectorQuantize(dim=768, codebook_size=int(self.pitch_lm.gpt_config.vocab_size * self.args['unsupervised_transcription_vq_codebook_size_factor']))

        # Initialize the output folder
        self.output_folder = f'{args["uglobals"]["OUTPUTS_DIR"]}/{args["task"]}/{args["name"]}'
        if args['mode'] == 'predict_dev':
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.output_idx = 0 # For indicing predictions

    # Optimization
    def configure_optimizers(self):
        params = [self.encoder.parameters(), self.decoder.parameters(), self.vq.parameters()]
        self.params_to_update = itertools.chain(*params)
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.args['lr'])
        # Since Adam is per-parameter, we don't need to re-initalize the optimizer when switching training modes

        # LR scheduler
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.args['lr_scheduler_start_factor'], end_factor=1, total_iters=self.args['lr_scheduler_warmup_epochs'])
        scheduler_anneal = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=self.args['lr_scheduler_end_factor'], total_iters=self.args['lr_scheduler_anneal_epochs'])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler_warmup, scheduler_anneal], milestones=[self.args['lr_scheduler_warmup_epochs']])
        return [optimizer], [scheduler]
    
    def on_before_optimizer_step(self, optimizer):
        # Track the gradient norms
        grad_norms = grad_norm(self, norm_type=2)['grad_2.0_norm_total']
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=1)
        self.log('train/grad_norms', grad_norms, batch_size=1)

    # Data Processing
    def preprocess(self, x):
        x = self.transform(x) # Wav to mel spectrogram
        x = 10 * torch.log10(x + 1e-6) # Log scale
        x = (x - self.args['uglobals']['SPECTORGRAM_MEAN']) / self.args['uglobals']['SPECTORGRAM_STD'] # Normalize
        x = x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]) # Merge the batch and seq_len dimensions, and add a channel dimension
        return x

    def postprocess(self, x, batch_size):
        x = x.reshape(batch_size, -1, 128, 16) # Split the batch and seq_len dimensions
        x = (x * self.args['uglobals']['SPECTORGRAM_STD']) + self.args['uglobals']['SPECTORGRAM_MEAN'] # Denormalize
        x = torch.pow(10, x / 10) - 1e-6 # Inverse log scale
        x = self.invers_transform(x) # Mel spectrogram to wav
        x = self.grifflim_transform(x)
        return x

    # Forward passes, losses and inference
    def get_pitch_lm_score(self, indices):
        return torch.ones_like(indices)[:, 0]
        # Zero pad
        indices = torch.nn.functional.pad(indices, (1, 0), value=0)
        
        # Query the pitch LM to get the sequence-level probabilities
        # TODO: Make this more efficient
        for i in range(indices.shape[1] - 1):
            logits = self.pitch_lm(indices[:, :i + 1])
            probs = torch.nn.functional.softmax(logits, dim=-1)[:, -1, :] # [batch, 128]

            indices_slice = indices[:, i + 1] # [batch]\
            probs = probs[range(probs.shape[0]), indices_slice]
            
            if i == 0:
                probs_total = probs
            else:
                probs_total *= probs
            
        return probs_total
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.squeeze()
        
        quantized, embed_ind, vq_loss = self.vq.forward_topk(x, self.args['unsupervised_transcription_vq_n_samples'], self.get_pitch_lm_score)
        # quantized, embed_ind, vq_loss = self.vq(x)

        # if no_vq:
            # quantized = x
            # embed_ind = torch.zeros_like(quantized)[:,:1]
            # vq_loss = torch.zeros(1)

        quantized = quantized.unsqueeze(-1).unsqueeze(-1)
        x_hat = self.decoder(quantized)
        x_hat = x_hat.reshape(x.shape[0], 1, 128, 16)
        return x_hat, quantized, embed_ind, vq_loss

    def batch_to_loss(self, batch, log_name, batch_idx):
        x = batch['wav']
        notes_target = batch['notes']
        batch_size = x.shape[0]
        x = self.preprocess(x)
        y = torch.clone(x)

        x_hat, quantized, embed_ind, vq_loss = self.forward(x)

        mse = torch.nn.MSELoss()(x_hat, y)
        loss = mse + self.args['unsupervised_transcription_vq_loss_weight'] * vq_loss
        accuracy = (embed_ind.reshape(-1) == notes_target.reshape(-1)).float().mean()

        self.log(f'{log_name}/loss', loss, batch_size=batch_size)
        self.log(f'{log_name}/mse', mse, batch_size=batch_size)
        self.log(f'{log_name}/vq_loss', vq_loss, batch_size=batch_size)
        self.log(f'{log_name}/accuracy', accuracy, batch_size=batch_size)
        self.log(f'{log_name}/monitor', accuracy, batch_size=batch_size) # Keep the best checkpoint based on this metric

        # if batch_idx == 0 and log_name=='val':
        #     print('VQ indices:\n', embed_ind.reshape(notes_target.shape)[:6])
        #     print('GT notes:\n', notes_target[:6])
        return loss

    def batch_to_infer(self, batch):
        x = batch['wav']
        notes_target = batch['notes']
        batch_size = x.shape[0]
        x = self.preprocess(x)
        y = torch.clone(x)

        x_hat, quantized, embed_ind, vq_loss = self.forward(x)

        return x_hat.detach(), y.detach(), batch['wav'], embed_ind.reshape(batch_size, -1), notes_target

    # Step functions
    def training_step(self, batch, batch_idx):
        loss = self.batch_to_loss(batch, 'train', batch_idx)
        return loss
    
    def eval_step(self, name, batch, batch_idx):
        loss = self.batch_to_loss(batch, name, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.eval_step('val', batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.eval_step('test', batch, batch_idx)
        return loss
 
    @torch.enable_grad() 
    def predict_step(self, batch, batch_idx):
        spectrogram_pred, spectrogram_target, wav_original, notes_pred, notes_target = self.batch_to_infer(batch)
        batch_size = wav_original.shape[0]

        wav_original = wav_original.reshape(batch_size, -1).cpu().numpy() #[batch, n_samples]
        wav_oracle = self.postprocess(spectrogram_target, batch_size).reshape(batch_size, -1).cpu().numpy()
        wav_pred = self.postprocess(spectrogram_pred, batch_size).reshape(batch_size, -1).cpu().numpy()

        names = batch['names']
        for i, name in enumerate(names):
            wav_write(f'{self.output_folder}/{batch_idx}_{i}_{name}_original.wav', self.sr, wav_original[i])
            wav_write(f'{self.output_folder}/{batch_idx}_{i}_{name}_oracle.wav',  self.sr, wav_oracle[i])
            wav_write(f'{self.output_folder}/{batch_idx}_{i}_{name}_pred.wav',  self.sr, wav_pred[i])
            with open(f'{self.output_folder}/{batch_idx}_{i}_{name}_notes.txt', 'w') as f:
                f.write('True:\n')
                f.write(str(notes_target[i]))
                f.write('\n')
                f.write('Pred:\n')
                f.write(str(notes_pred[i]))
        return