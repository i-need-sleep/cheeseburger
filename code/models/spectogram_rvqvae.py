from pathlib import Path
from copy import deepcopy

from scipy.io.wavfile import write as wav_write
import torch, torchvision, torchaudio
import lightning
import vector_quantize_pytorch

from models.modeling_utils.res_module import ResModule

class Spectorgram_RVQVAE(lightning.LightningModule):
    def __init__(self, args, sr):
        super().__init__()
        self.args = args
        self.sr = sr
        self.save_hyperparameters(args, sr)

        # Transformations
        n_fft = self.args['uglobals']['N_FFT']
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft)
        self.invers_transform = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=n_stft)
        self.grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft)

        # Modeling
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
        self.rvq = vector_quantize_pytorch.ResidualVQ(
            dim = 768,
            num_quantizers = 8,
            codebook_size = 1024,
            stochastic_sample_codes = True,
            sample_codebook_temp = 0.1,
            shared_codebook = True
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

        # Initialize the output folder
        self.output_folder = f'{args["uglobals"]["OUTPUTS_DIR"]}/{args["task"]}/{args["name"]}'
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.output_idx = 0 # For indicing predictions

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

    def forward(self, x):
        # [batch_size, seq_len, 128, 16]
        x = self.encoder(x)
        x = torch.squeeze(x)

        quantized, indices, commit_loss = self.rvq(x)
        # [batch_size, seq_len, 768], [batch_size, seq_len, n_quantizers], [batch_size, n_quantizers]
        commit_loss = torch.mean(commit_loss)
        
        quantized = quantized.unsqueeze(-1).unsqueeze(-1)
        x_hat = self.decoder(quantized)
        
        return x_hat, quantized, indices, commit_loss

    def training_step(self, batch, batch_idx):
        x = batch['wav']
        batch_size = x.shape[0]
        x = self.preprocess(x)
        y = deepcopy(x)
        y_hat, _, _, commit_loss = self.forward(x)
        recons_loss = torch.nn.functional.mse_loss(y_hat, y)
        commit = self.args['commit_loss_weight'] * commit_loss
        loss = recons_loss + commit_loss

        self.log("train/loss", loss, batch_size=batch_size)
        self.log("train/recons_loss", recons_loss, batch_size=batch_size)
        self.log("train/commit_loss", commit_loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['wav']
        batch_size = x.shape[0]
        x = self.preprocess(x)
        y = deepcopy(x)
        y_hat, _, _, commit_loss = self.forward(x)
        recons_loss = torch.nn.functional.mse_loss(y_hat, y)
        loss = recons_loss + commit_loss

        self.log("val/loss", loss, batch_size=batch_size) # Automatically averaged
        self.log("val/recons_loss", recons_loss, batch_size=batch_size)
        self.log("val/commit_loss", commit_loss, batch_size=batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['wav']
        batch_size = x.shape[0]
        x = self.preprocess(x)
        y = deepcopy(x)
        y_hat, _, _, commit_loss = self.forward(x)
        recons_loss = torch.nn.functional.mse_loss(y_hat, y)
        loss = recons_loss + commit_loss

        self.log("test/loss", loss, batch_size=batch_size) # Automatically averaged
        self.log("test/recons_loss", recons_loss, batch_size=batch_size)
        self.log("test/commit_loss", commit_loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        return optimizer
    
    @torch.enable_grad() 
    def predict_step(self, batch, batch_idx):
        if self.output_idx >= self.args['n_predictions']:
            return
        x = batch['wav']
        names = batch['names']
        batch_size = x.shape[0]
        x_original = deepcopy(x).reshape(batch_size, -1) # Original wav
        x = self.preprocess(x)
        x_processed = deepcopy(x) # Mel spectrogram
        x_recons, _, _, c = self.forward(x)
        
        x_processed = self.postprocess(x_processed, batch_size).reshape(batch_size, -1)
        x_recons = self.postprocess(x_recons.detach(), batch_size).reshape(batch_size, -1)

        for i, name in enumerate(names):
            wav_write(f'{self.output_folder}/{name}_original.wav', self.sr, x_original[i].cpu().numpy())
            wav_write(f'{self.output_folder}/{name}_processed.wav',  self.sr, x_processed[i].cpu().numpy())
            wav_write(f'{self.output_folder}/{name}_recons.wav',  self.sr, x_recons[i].cpu().numpy())
            self.output_idx += 1
        
            if self.output_idx >= self.args['n_predictions']:
                return
        return 