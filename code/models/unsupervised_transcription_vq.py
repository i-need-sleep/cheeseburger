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
        # self.vq = RerankVQ(dim=768, codebook_size=self.pitch_lm.gpt_config.vocab_size)
        self.vq = VectorQuantize(dim=768, codebook_size=self.pitch_lm.gpt_config.vocab_size)

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
        
        # quantized, embed_ind, vq_loss = self.rerank_vq.forward_topk(x, self.args['unsupervised_transcription_vq_n_samples'], self.get_pitch_lm_score)
        quantized, embed_ind, vq_loss = self.vq(x)

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
        self.log(f'{log_name}/monitor', loss, batch_size=batch_size)
        self.log(f'{log_name}/accuracy', accuracy, batch_size=batch_size)
        self.log(f'{log_name}/monitor', accuracy, batch_size=batch_size) # Keep the best checkpoint based on this metric

        # if batch_idx == 0 and log_name=='val':
        #     print('VQ indices:\n', embed_ind.reshape(notes_target.shape)[:6])
        #     print('GT notes:\n', notes_target[:6])
        return loss

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
    
# class UnsupervisedTranscriptionVQ(lightning.LightningModule):
#     def __init__(self, args, sr):
#         super().__init__()
#         self.args = args
#         self.sr = sr
#         self.save_hyperparameters()
        
#         # Modeling: Pretrained, frozen models
#         print(f'Loading WAV Model from checkpoint: {args["det_cheese_wav_lm_checkpoint"]}')
#         self.wav_lm = DeterministicWavTransformer.load_from_checkpoint(args["det_cheese_wav_lm_checkpoint"])

#         print(f'Loading Pitch LM from checkpoint: {args["det_cheese_pitch_lm_checkpoint"]}')
#         self.pitch_lm = PitchLM.load_from_checkpoint(args["det_cheese_pitch_lm_checkpoint"])

#         self.pitch_lm.freeze()

#         # Modeling: Custom VQ layer
#         self.rerank_vq = RerankVQ(dim=self.wav_lm.gpt_config.n_embd, codebook_size=self.pitch_lm.gpt_config.vocab_size)
        
#         # Modeling: Encoder/Decoder
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(2, 2)),
#             torch.nn.ReLU(),
#             ResModule(
#                 torch.nn.Sequential(
#                     torch.nn.Conv2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                     torch.nn.Conv2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                 )
#             ),ResModule(
#                 torch.nn.Sequential(
#                     torch.nn.Conv2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                     torch.nn.Conv2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                 )
#             ),
#             torch.nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(2, 2)),
#             torch.nn.ReLU(),
#             ResModule(
#                 torch.nn.Sequential(
#                     torch.nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                     torch.nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                 )
#             ),
#             ResModule(
#                 torch.nn.Sequential(
#                     torch.nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                     torch.nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                 )
#             ),
#             torch.nn.Conv2d(16, 768, kernel_size=(128, 16)),
#         )
#         self.decoder = torch.nn.Sequential(
#             torch.nn.ConvTranspose2d(768, 16, kernel_size=(128, 16)),
#             torch.nn.ReLU(),
#             ResModule(
#                 torch.nn.Sequential(
#                     torch.nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                     torch.nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                 )
#             ),
#             ResModule(
#                 torch.nn.Sequential(
#                     torch.nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                     torch.nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                 )
#             ),
#             torch.nn.ConvTranspose2d(16, 8, kernel_size=(5, 5), padding=(2, 2)),
#             torch.nn.ReLU(),
#             ResModule(
#                 torch.nn.Sequential(
#                     torch.nn.ConvTranspose2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                     torch.nn.ConvTranspose2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                 )
#             ),
#             ResModule(
#                 torch.nn.Sequential(
#                     torch.nn.ConvTranspose2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                     torch.nn.ConvTranspose2d(8, 8, kernel_size=(5, 5), padding=(2, 2)),
#                     torch.nn.ReLU(),
#                 )
#             ),
#             torch.nn.ConvTranspose2d(8, 1, kernel_size=(5, 5), padding=(2, 2)),
#         )

#         # Output
#         self.output_folder = f'{args["uglobals"]["OUTPUTS_DIR"]}/{args["task"]}/{args["name"]}'
#         if args['mode'] == 'predict_dev':
#             Path(self.output_folder).mkdir(parents=True, exist_ok=True)

#     # Optimization
#     def configure_optimizers(self):
#         params = [self.encoder.parameters(), self.decoder.parameters(), self.rerank_vq.parameters()]
#         self.params_to_update = itertools.chain(*params)
#         optimizer = torch.optim.Adam(self.params_to_update, lr=self.args['lr'])
#         # Since Adam is per-parameter, we don't need to re-initalize the optimizer when switching training modes

#         # LR scheduler
#         scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.args['lr_scheduler_start_factor'], end_factor=1, total_iters=self.args['lr_scheduler_warmup_epochs'])
#         scheduler_anneal = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=self.args['lr_scheduler_end_factor'], total_iters=self.args['lr_scheduler_anneal_epochs'])
#         scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler_warmup, scheduler_anneal], milestones=[self.args['lr_scheduler_warmup_epochs']])
#         return [optimizer], [scheduler]
    
#     def on_before_optimizer_step(self, optimizer):
#         # Track the gradient norms
#         grad_norms = grad_norm(self, norm_type=2)['grad_2.0_norm_total']
#         self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=1)
#         self.log('train/grad_norms', grad_norms, batch_size=1)

#     # Data Processing
#     def prep_batch(self, batch):
#         wav = batch['wav']
#         notes = batch['notes']

#         batch_size = wav.shape[0]
#         seq_len = wav.shape[1]
        
#         spectrogram_target = self.wav_lm.preprocess_wav(wav) # [batch_size, seq_len, 128, 16]
#         spectrogram_in = spectrogram_target[:, : -1] # Leave space for BoS

#         notes_in, notes_target = self.pitch_lm.preprocess(notes) # Zero-padded and shifted: [batch_size, seq_len]
#         return wav, spectrogram_in, spectrogram_target, notes_in, notes_target, batch_size, seq_len

#     # Forward passes, losses and inference
#     def get_pitch_lm_score(self, indices):
#         # Zero pad
#         indices = torch.nn.functional.pad(indices, (1, 0), value=0)
        
#         # Query the pitch LM to get the sequence-level probabilities
#         # TODO: Make this more efficient
#         for i in range(indices.shape[1] - 1):
#             logits = self.pitch_lm(indices[:, :i + 1])
#             probs = torch.nn.functional.softmax(logits, dim=-1)[:, -1, :] # [batch, 128]

#             indices_slice = indices[:, i + 1] # [batch]\
#             probs = probs[range(probs.shape[0]), indices_slice]
            
#             if i == 0:
#                 probs_total = probs
#             else:
#                 probs_total *= probs
            
#         return probs_total

#     def joint_forward(self, x):
#         # x: spectrogram [batch_size, seq_len, 128, 16]
#         batch_size = x.shape[0]
#         seq_len = x.shape[1]

#         # Encode
#         print(x.shape)
#         x = x.reshape(batch_size * seq_len, 1, x.shape[2], x.shape[3]) # [batch_size * (seq_len-1), 1, 128, 16]
#         x = self.wav_lm.encoder(x)
#         x = x.reshape(batch_size, seq_len, -1) # [batch_size, seq_len-1, emb_size]
        
#         # Wav model before adaptor
#         x = self.wav_lm.gpt.forward_before(inputs_embeds=x)

#         # Pitch branch
#         quantized, embed_ind, vq_loss = self.rerank_vq.forward_topk(x, self.args['unsupervised_transcription_vq_n_samples'], self.get_pitch_lm_score)

#         x = self.wav_lm.gpt.forward_after(overwrite_hidden_states=quantized, inputs_embeds=quantized).last_hidden_state
        
#         # Decode
#         x = x.reshape(batch_size * seq_len, -1, 1, 1) # [batch_size * seq_len, emb_size, 1, 1]
#         x = self.wav_lm.decoder(x)
#         x = x.reshape(batch_size, seq_len, x.shape[2], x.shape[3]) # [batch_size, seq_len, 128, 16]
#         return x, embed_ind, vq_loss
    
#     def joint_loss_and_log(self, spectrogram_pred, spectrogram_target, embed_ind, notes_target, vq_loss, batch_size, log_name, print_out=False):
#         mse = torch.nn.MSELoss()(spectrogram_pred, spectrogram_target)
#         loss = mse + self.args['unsupervised_transcription_vq_loss_weight'] * vq_loss
#         accuracy = (embed_ind.reshape(-1) == notes_target.reshape(-1)).float().mean()

#         self.log(f'{log_name}/loss', loss, batch_size=batch_size)
#         self.log(f'{log_name}/mse', mse, batch_size=batch_size)
#         self.log(f'{log_name}/vq_loss', vq_loss, batch_size=batch_size)
#         self.log(f'{log_name}/accuracy', accuracy, batch_size=batch_size)
#         self.log(f'{log_name}/monitor', accuracy, batch_size=batch_size) # Keep the best checkpoint based on this metric

#         if print_out:
#             print('VQ indices:\n', embed_ind[:6])
#             print('GT notes:\n', notes_target[:6])

#         return loss

#     def batch_to_loss(self, batch, name, batch_idx):
#         wav, spectrogram_in, spectrogram_target, notes_in, notes_target, batch_size, seq_len = self.prep_batch(batch)

#         spectrogram_pred, embed_ind, vq_loss = self.joint_forward(spectrogram_target)
#         loss = self.joint_loss_and_log(spectrogram_pred, spectrogram_target, embed_ind, notes_target, vq_loss, batch_size, name, print_out=batch_idx==0)
#         return loss
    
#     # Step functions
#     def training_step(self, batch, batch_idx):
#         loss = self.batch_to_loss(batch, 'train', batch_idx)
#         return loss
    
#     def eval_step(self, name, batch, batch_idx):
#         loss = self.batch_to_loss(batch, name, batch_idx)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss = self.eval_step('val', batch, batch_idx)
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         loss = self.eval_step('test', batch, batch_idx)
#         return loss
