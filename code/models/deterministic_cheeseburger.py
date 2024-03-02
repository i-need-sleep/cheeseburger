from pathlib import Path
import types
import itertools
from copy import deepcopy
from scipy.io.wavfile import write as wav_write

import torch
import torchaudio
import lightning
from lightning.pytorch.utilities import grad_norm
import transformers

from models.deterministic_wav_transformer import DeterministicWavTransformer
from models.pitch_lm import PitchLM
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
    
class DeterministicCheeseburger(lightning.LightningModule):
    def __init__(self, args, sr):
        super().__init__()
        self.args = args
        self.sr = sr
        self.save_hyperparameters()
        
        # Modeling: Pretrained, frozen models
        print(f'Loading WAV Model from checkpoint: {args["det_cheese_wav_lm_checkpoint"]}')
        self.wav_lm = DeterministicWavTransformer.load_from_checkpoint(args["det_cheese_wav_lm_checkpoint"])

        print(f'Loading Pitch LM from checkpoint: {args["det_cheese_pitch_lm_checkpoint"]}')
        self.pitch_lm = PitchLM.load_from_checkpoint(args["det_cheese_pitch_lm_checkpoint"])

        self.wav_lm.freeze()
        self.pitch_lm.freeze()

        # Modeling: Adaptors
        self.adaptor_in = Adaptor(self.wav_lm.gpt_config.n_embd, self.pitch_lm.gpt_config.n_embd)
        self.adaptor_out = Adaptor(self.pitch_lm.gpt_config.vocab_size, self.wav_lm.gpt_config.n_embd)
        self.adaptor_skip = Adaptor(self.wav_lm.gpt_config.n_embd, self.wav_lm.gpt_config.n_embd)
        
        # Patch the wav lm
        self.wav_lm.gpt.det_cheese_insertion_layer = args['det_cheese_insertion_layer']
        self.wav_lm.gpt.forward_before = types.MethodType(gpt_patch.patched_forward_before, self.wav_lm.gpt)
        self.wav_lm.gpt.forward_after = types.MethodType(gpt_patch.patched_forward_after, self.wav_lm.gpt)

        # Output
        self.output_folder = f'{args["uglobals"]["OUTPUTS_DIR"]}/{args["task"]}/{args["name"]}'
        if args['mode'] == 'predict_dev':
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    # Optimization
    def configure_optimizers(self):
        params = [self.adaptor_in.parameters(), self.adaptor_out.parameters(), self.adaptor_skip.parameters()]
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
    def prep_batch(self, batch):
        wav = batch['wav']
        notes = batch['notes']

        batch_size = wav.shape[0]
        seq_len = wav.shape[1]
        
        spectrogram_target = self.wav_lm.preprocess_wav(wav) # [batch_size, seq_len, 128, 16]
        spectrogram_in = spectrogram_target[:, : -1] # Leave space for BoS

        notes_in, notes_target = self.pitch_lm.preprocess(notes) # Zero-padded and shifted: [batch_size, seq_len]
        return wav, spectrogram_in, spectrogram_target, notes_in, notes_target, batch_size, seq_len

    # Forward passes, losses and inference
    def joint_forward(self, x, skip_only=False):
        # x: spectrogram [batch_size, seq_len-1, 128, 16]
        batch_size = x.shape[0]
        seq_len = x.shape[1] + 1

        # Encode
        x = x.reshape(batch_size * (seq_len - 1), 1, x.shape[2], x.shape[3]) # [batch_size * (seq_len-1), 1, 128, 16]
        x = self.wav_lm.encoder(x)
        x = x.reshape(batch_size, seq_len - 1, -1) # [batch_size, seq_len-1, emb_size]

        # BoS
        bos = self.wav_lm.get_bos_emb(batch_size)
        x = torch.cat([bos, x], dim=1) # [batch_size, seq_len, emb_size], pad the BoS token
        
        # Wav model before adaptor
        x = self.wav_lm.gpt.forward_before(inputs_embeds=x)

        # Skip branch
        h_skip = self.adaptor_skip(x)

        # Pitch branch
        x = self.adaptor_in(x)
        x = self.pitch_lm.gpt(inputs_embeds=x).last_hidden_state
        notes_logits = self.pitch_lm.lm_head(x)

        if self.intervention_mode != '':
            print(self.intervention_mode, self.intervention_step)

            if self.intervention_mode == 'sample_patch':
                notes_logits[:] = notes_logits[0]
            else:
                for i in range(notes_logits.shape[0]):

                    if self.intervention_step == 'all':
                        range_start = 0
                        range_end = notes_logits.shape[1]
                    elif self.intervention_step == 'last':
                        range_start = notes_logits.shape[1] - 1
                        range_end = notes_logits.shape[1]
                    else:
                        raise NotImplementedError
                    
                    for j in range(range_start, range_end):
                        if self.intervention_mode == 'swap':
                            val = deepcopy(notes_logits[i, j, 60].detach())
                            max_idx = notes_logits[i, j, :].argmax()
                            notes_logits[i, j, 60] = torch.max(notes_logits[i, j, :])
                            notes_logits[i, j, max_idx] = val
                        elif self.intervention_mode == '01':
                            notes_logits = torch.zeros_like(notes_logits)
                            notes_logits[i, j, 60] = 1
                        else:
                            raise NotImplementedError

        h_pitch = self.adaptor_out(notes_logits)

        if skip_only:
            # Train only the skip adaptor
            h_pitch = h_pitch.detach()

        # Merged branch
        x = h_pitch + h_skip
        x = self.wav_lm.gpt.forward_after(overwrite_hidden_states=x, inputs_embeds=x).last_hidden_state
        
        # Decode
        x = x.reshape(batch_size * seq_len, -1, 1, 1) # [batch_size * seq_len, emb_size, 1, 1]
        x = self.wav_lm.decoder(x)
        x = x.reshape(batch_size, seq_len, x.shape[2], x.shape[3]) # [batch_size, seq_len, 128, 16]
        return x, notes_logits
    
    def joint_loss_and_log(self, spectrogram_pred, spectrogram_target, notes_logits, notes_target, batch_size, log_name, skip_only=False):
        mse = torch.nn.MSELoss()(spectrogram_pred, spectrogram_target)
        ce = torch.nn.CrossEntropyLoss()(notes_logits.reshape(-1, notes_logits.shape[-1]), notes_target.reshape(-1))
        loss = mse + self.args['det_cheese_ce_weight'] * ce
        accuracy = (notes_logits.argmax(-1) == notes_target).float().mean()

        self.log(f'{log_name}/loss', loss, batch_size=batch_size)
        self.log(f'{log_name}/monitor', loss, batch_size=batch_size) # Keep the best checkpoint based on this metric
        self.log(f'{log_name}/mse', mse, batch_size=batch_size)
        self.log(f'{log_name}/ce', ce, batch_size=batch_size)
        self.log(f'{log_name}/accuracy', accuracy, batch_size=batch_size)

        training_mode = 0
        if skip_only:
            training_mode = 3
        self.log(f'{log_name}/training_mode', training_mode, batch_size=batch_size)
        return loss

    def pre_branch_forward(self, x):
        # x: spectrogram [batch_size, seq_len-1, 128, 16]
        batch_size = x.shape[0]
        seq_len = x.shape[1] + 1

        # Encode
        x = x.reshape(batch_size * (seq_len - 1), 1, x.shape[2], x.shape[3]) # [batch_size * (seq_len-1), 1, 128, 16]
        x = self.wav_lm.encoder(x)
        x = x.reshape(batch_size, seq_len - 1, -1) # [batch_size, seq_len-1, emb_size]

        # BoS
        bos = self.wav_lm.get_bos_emb(batch_size)
        x = torch.cat([bos, x], dim=1) # [batch_size, seq_len, emb_size], pad the BoS token
        
        # Wav model before adaptor
        x = self.wav_lm.gpt.forward_before(inputs_embeds=x)

        # Pitch branch
        x = self.adaptor_in(x)
        x = self.pitch_lm.gpt(inputs_embeds=x).last_hidden_state
        notes_logits = self.pitch_lm.lm_head(x)
        return notes_logits 

    def pre_branch_loss_and_log(self, notes_logits, notes_target, batch_size, log_name):
        ce = torch.nn.CrossEntropyLoss()(notes_logits.reshape(-1, notes_logits.shape[-1]), notes_target.reshape(-1))
        accuracy = (notes_logits.argmax(-1) == notes_target).float().mean()

        self.log(f'{log_name}/ce', ce, batch_size=batch_size)
        self.log(f'{log_name}/monitor', -1 * accuracy, batch_size=batch_size) # Keep the best checkpoint based on this metric
        self.log(f'{log_name}/accuracy', accuracy, batch_size=batch_size)

        self.log(f'{log_name}/training_mode', 1, batch_size=batch_size)
        return ce
    
    def post_brach_forward(self, notes):
        batch_size = notes.shape[0]
        seq_len = notes.shape[1]

        x = self.pitch_lm.gpt(notes).last_hidden_state
        x = self.pitch_lm.lm_head(x)
        x = self.adaptor_out(x)

        x = self.wav_lm.gpt.forward_after(overwrite_hidden_states=x, inputs_embeds=x).last_hidden_state
        
        # Decode
        x = x.reshape(batch_size * seq_len, -1, 1, 1) # [batch_size * seq_len, emb_size, 1, 1]
        x = self.wav_lm.decoder(x)
        x = x.reshape(batch_size, seq_len, x.shape[2], x.shape[3]) # [batch_size, seq_len, 128, 16]
        return x
    
    def post_branch_loss_and_log(self, spectrogram_pred, spectrogram_target, batch_size, log_name):
        mse = torch.nn.MSELoss()(spectrogram_pred, spectrogram_target)

        self.log(f'{log_name}/mse', mse, batch_size=batch_size)
        self.log(f'{log_name}/monitor', mse, batch_size=batch_size) # Keep the best checkpoint based on this metric
        self.log(f'{log_name}/training_mode', 2, batch_size=batch_size)
        return mse

    def batch_to_loss(self, batch, name):
        wav, spectrogram_in, spectrogram_target, notes_in, notes_target, batch_size, seq_len = self.prep_batch(batch)

        if self.training_mode == 'joint':
            spectrogram_pred, notes_logits = self.joint_forward(spectrogram_in)
            loss = self.joint_loss_and_log(spectrogram_pred, spectrogram_target, notes_logits, notes_target, batch_size, name)
        elif self.training_mode == 'pre_branch':
            notes_logits = self.pre_branch_forward(spectrogram_in)
            loss = self.pre_branch_loss_and_log(notes_logits, notes_target, batch_size, name)
        elif self.training_mode == 'post_branch':
            spectrogram_pred = self.post_brach_forward(notes_in)
            loss = self.post_branch_loss_and_log(spectrogram_pred, spectrogram_target, batch_size, name)
        elif self.training_mode == 'skip_branch':
            spectrogram_pred, notes_logits = self.joint_forward(spectrogram_in, skip_only=True)
            loss = self.joint_loss_and_log(spectrogram_pred, spectrogram_target, notes_logits, notes_target, batch_size, name, skip_only=True)
        else:
            raise NotImplementedError
        return loss
    
    def infer(self, spectrogram_in, notes_in):
        seq_len = spectrogram_in.shape[1] + 1
        spectrogram_in = spectrogram_in[:, :self.test_context_len]
        notes_in = notes_in[:, 1: 1 + self.test_context_len] # Remove the BoS token
        
        while spectrogram_in.shape[1] < seq_len:
            spectrogram_pred, notes_logits = self.joint_forward(deepcopy(spectrogram_in.detach()))
            notes_pred = notes_logits.argmax(-1)
            spectrogram_in = torch.cat([spectrogram_in, spectrogram_pred[:, -1:]], dim=1).detach()
            notes_in = torch.cat([notes_in, notes_pred[:, -1:]], dim=1).detach()
            
        return spectrogram_in, notes_in
    
    # Step functions
    def training_step(self, batch, batch_idx):
        loss = self.batch_to_loss(batch, 'train')
        return loss
    
    def eval_step(self, name, batch, batch_idx):
        loss = self.batch_to_loss(batch, name)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.eval_step('val', batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.eval_step('test', batch, batch_idx)
        return loss

    @torch.enable_grad() 
    def predict_step(self, batch, batch_idx):
        wav, spectrogram_in, spectrogram_target, notes_in, notes_target, batch_size, seq_len = self.prep_batch(batch)
        spectrogram_pred, notes_pred = self.infer(spectrogram_in, notes_in)

        wav_original = wav.reshape(batch_size, -1).cpu().numpy() #[batch, n_samples]
        wav_oracle = self.wav_lm.postprocess_wav(spectrogram_target, batch_size).reshape(batch_size, -1).cpu().numpy()
        wav_pred = self.wav_lm.postprocess_wav(spectrogram_pred, batch_size).reshape(batch_size, -1).cpu().numpy()

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
        exit()
        return
