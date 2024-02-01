from pathlib import Path

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
        print(f'loading spectrogram AE checkpoint: {args["rvqvae_checkpoint"]}')
        # ckpt = torch.load(args['rvqvae_checkpoint'], map_location='cpu')
        # print(ckpt.keys())
        # exit()
        self.rvqvae = Spectorgram_RVQVAE.load_from_checkpoint(args['rvqvae_checkpoint'], )
        self.gpt_config = self.get_gpt_config()
        self.gpt = transformers.AutoModel.from_config(self.gpt_config)
        self.bos_emb = torch.nn.Embedding(1, self.gpt_config.n_embd)
        exit()

    def get_gpt_config(self):
        # Modify the config of a distill-gpt2 model
        config = transformers.AutoConfig.from_pretrained('distilgpt2')
        return config

    def forward(self, x):
        return

    def training_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        return optimizer
    
