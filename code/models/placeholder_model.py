from pathlib import Path

import torch, torchvision
import lightning
import vector_quantize_pytorch

from models.modeling_utils.res_module import ResModule

class PlaceholderModel(lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(5, 5), padding=(2, 2)),
            torch.nn.ReLU(),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            torch.nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            torch.nn.ReLU(),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            torch.nn.Conv2d(64, 128, kernel_size=(28, 28)),
        )
        self.rvq = vector_quantize_pytorch.ResidualVQ(
            dim = 128,
            num_quantizers = 8,
            codebook_size = 512,
            stochastic_sample_codes = True,
            sample_codebook_temp = 0.1,
            shared_codebook = True
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=(28, 28)),
            torch.nn.ReLU(),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), padding=(2, 2)),
            torch.nn.ReLU(),
            ResModule(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), padding=(2, 2)),
                    torch.nn.ReLU(),
                )
            ),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=(5, 5), padding=(2, 2)),
        )

        # Initialize the output folder
        self.output_folder = f'{args["uglobals"]["OUTPUTS_DIR"]}/{args["task"]}/{args["name"]}'
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.output_idx = 0 # For indicing predictions

    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        x = self.encoder(x)
        x = torch.squeeze(x)

        quantized, indices, commit_loss = self.rvq(x)
        # [batch_size, 128], [batch_size, n_quantizers], [batch_size, n_quantizers]
        commit_loss = torch.mean(commit_loss)
        
        quantized = quantized.unsqueeze(-1).unsqueeze(-1)
        x_hat = self.decoder(quantized)
        return x_hat, commit_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch # Discard the labels. Train on an autoencoding task.
        x_hat, commit_loss = self.forward(x)
        recons_loss = torch.nn.functional.mse_loss(x_hat, x)
        loss = recons_loss + commit_loss

        self.log("train/loss", loss)
        self.log("train/recons_loss", recons_loss)
        self.log("train/commit_loss", commit_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, commit_loss = self.forward(x)
        recons_loss = torch.nn.functional.mse_loss(x_hat, x)
        loss = recons_loss + commit_loss

        self.log("val/loss", loss) # Automatically averaged
        self.log("val/recons_loss", recons_loss)
        self.log("val/commit_loss", commit_loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, commit_loss = self.forward(x)
        recons_loss = torch.nn.functional.mse_loss(x_hat, x)
        loss = recons_loss + commit_loss

        self.log("val/loss", loss) # Automatically averaged
        self.log("val/recons_loss", recons_loss)
        self.log("val/commit_loss", commit_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        return optimizer
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, commit_loss = self.forward(x)
        
        # Save the predictions/GT
        for i in range(x.shape[0]):
            torchvision.utils.save_image(x[i], f'{self.output_folder}/{self.output_idx}_gt.png')
            torchvision.utils.save_image(x_hat[i], f'{self.output_folder}/{self.output_idx}_pred.png')
            self.output_idx += 1
            if self.output_idx >= self.args['n_predictions']:
                return
        return 