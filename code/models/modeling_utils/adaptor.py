import torch
import transformers

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