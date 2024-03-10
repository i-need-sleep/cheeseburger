import itertools
from copy import deepcopy

import torch

from models.deterministic_cheeseburger import DeterministicCheeseburger
from models.modeling_utils.adaptor import Adaptor
from models.modeling_utils.grad_reverse import grad_reverse
from models.modeling_utils.misc import off_diagonal

class DeterministicCheeseburgerUnsupervised(DeterministicCheeseburger):
    def __init__(self, args, sr):
        super().__init__(args, sr)
        self.pitch_adversary = Adaptor(self.pitch_lm.gpt_config.vocab_size, 2)

    # Optimization
    def configure_optimizers(self):
        params = [self.adaptor_in.parameters(), self.adaptor_out.parameters(), self.adaptor_skip.parameters(), self.pitch_adversary.parameters()]
        self.params_to_update = itertools.chain(*params)
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.args['lr'])
        # Since Adam is per-parameter, we don't need to re-initalize the optimizer when switching training modes

        # LR scheduler
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.args['lr_scheduler_start_factor'], end_factor=1, total_iters=self.args['lr_scheduler_warmup_epochs'])
        scheduler_anneal = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=self.args['lr_scheduler_end_factor'], total_iters=self.args['lr_scheduler_anneal_epochs'])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler_warmup, scheduler_anneal], milestones=[self.args['lr_scheduler_warmup_epochs']])
        return [optimizer], [scheduler]
    
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

        # Post intervention
        h_pitch = notes_logits
        if self.args['det_cheese_softmax_logits']:
            h_pitch = torch.nn.functional.softmax(h_pitch, dim=-1)
        h_pitch = self.adaptor_out(h_pitch)

        if skip_only:
            # Train only the skip adaptor
            h_pitch = h_pitch.detach()
            notes_logits = notes_logits.detach()

        # Merged branch
        x = h_pitch + h_skip
        x = self.wav_lm.gpt.forward_after(overwrite_hidden_states=x, inputs_embeds=x).last_hidden_state
        
        # Decode
        x = x.reshape(batch_size * seq_len, -1, 1, 1) # [batch_size * seq_len, emb_size, 1, 1]
        x = self.wav_lm.decoder(x)
        x = x.reshape(batch_size, seq_len, x.shape[2], x.shape[3]) # [batch_size, seq_len, 128, 16]
        return x, notes_logits, h_skip
    
    def adversary_forward(self, logits_pred, notes_in):
        logits_true = self.pitch_lm.gpt(notes_in).last_hidden_state
        logits_true = self.pitch_lm.lm_head(logits_true)

        logits_all = torch.cat([logits_pred, logits_true], dim=0)
        labels_all = torch.cat([torch.zeros(logits_pred.shape[0]), torch.ones(logits_true.shape[0])], dim=0).long().to(logits_pred.device)
        logits_all = grad_reverse(logits_all)
        adversary_pred = self.pitch_adversary(logits_all)[:, 0, :] # [batch_size, 2]
        return adversary_pred, labels_all
    
    def vc_loss(self, h_skip):
        h_skip = h_skip.reshape(-1, h_skip.shape[-1]) # Also penalize the covariance between time steps
        h_skip = h_skip - h_skip.mean(dim=0)

        std = torch.mean(torch.sqrt(h_skip.var(dim=0) + 0.0001))

        cov = (h_skip.T @ h_skip) / (h_skip.shape[0] - 1)
        cov = off_diagonal(cov).pow_(2).sum().div(h_skip.shape[1]) 
        return std, cov
    
    def joint_loss_and_log(self, spectrogram_pred, spectrogram_target, adversary_pred, adversary_labels, h_skip, spectrogram_size, notes_size, log_name, skip_only=False):

        mse = torch.nn.MSELoss()(spectrogram_pred, spectrogram_target)
        adv_loss = torch.nn.CrossEntropyLoss()(adversary_pred.reshape(-1, adversary_pred.shape[-1]), adversary_labels.reshape(-1))
        std, cov = self.vc_loss(h_skip)

        loss = mse \
            + self.args['det_cheese_unsup_adv_weight'] * adv_loss \
            + self.args['det_cheese_unsup_std_weight'] * std \
            + self.args['det_cheese_unsup_cov_weight'] * torch.nn.functional.relu(self.args['det_cheese_unsup_cov_margin'] - cov) # Encourage the covariance to be large
        
        batch_size = spectrogram_size + notes_size

        self.log(f'{log_name}/loss', loss, batch_size=batch_size)
        self.log(f'{log_name}/monitor', loss, batch_size=batch_size) # Keep the best checkpoint based on this metric
        self.log(f'{log_name}/mse_out', mse, batch_size=batch_size)
        self.log(f'{log_name}/adv_loss', adv_loss, batch_size=batch_size)
        self.log(f'{log_name}/std_loss', std, batch_size=batch_size)
        self.log(f'{log_name}/cov_loss', cov, batch_size=batch_size)

        training_mode = 0
        if skip_only:
            training_mode = 3
        self.log(f'{log_name}/training_mode', training_mode, batch_size=batch_size)
        return loss
    
    def batch_to_loss(self, batch, name):
        wav, spectrogram_in, spectrogram_target, notes_in, notes_target, batch_size, seq_len = self.prep_batch(batch)

        if self.training_mode == 'joint':
            # Split the batch
            spectrogram_size = batch_size // 10 # EDIT ME
            notes_size = batch_size - spectrogram_size
            spectrogram_in = spectrogram_in[:spectrogram_size]
            spectrogram_target = spectrogram_target[:spectrogram_size]
            notes_in = notes_in[spectrogram_size:]
            notes_target = notes_target[spectrogram_size:]

            # Forward pass
            spectrogram_pred, notes_logits, h_skip = self.joint_forward(spectrogram_in)
            # Adversary forward pass
            adversary_pred, adversary_labels = self.adversary_forward(notes_logits, notes_in)

            loss = self.joint_loss_and_log(spectrogram_pred, spectrogram_target, adversary_pred, adversary_labels, h_skip, spectrogram_size, notes_size, name)
        else:
            raise NotImplementedError
        return loss
    
    def infer(self, spectrogram_in, notes_in):
        seq_len = spectrogram_in.shape[1] + 1
        spectrogram_in = spectrogram_in[:, :self.test_context_len]
        notes_in = notes_in[:, 1: 1 + self.test_context_len] # Remove the BoS token
        
        while spectrogram_in.shape[1] < seq_len:
            spectrogram_pred, notes_logits = self.joint_forward_plain(deepcopy(spectrogram_in.detach()))
            notes_pred = notes_logits.argmax(-1)
            spectrogram_in = torch.cat([spectrogram_in, spectrogram_pred[:, -1:]], dim=1).detach()
            notes_in = torch.cat([notes_in, notes_pred[:, -1:]], dim=1).detach()
            
        return spectrogram_in, notes_in
    
    def joint_forward_plain(self, x, skip_only=False):
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
        if not self.args['det_cheese_aug_x_no_pitch_model']:
            x = self.pitch_lm.gpt(inputs_embeds=x).last_hidden_state
        notes_logits = self.pitch_lm.lm_head(x)

        # Intervention
        notes_logits = self.apply_intervention(notes_logits)

        # Post intervention
        h_pitch = notes_logits
        if self.args['det_cheese_softmax_logits']:
            h_pitch = torch.nn.functional.softmax(h_pitch, dim=-1)
        h_pitch = self.adaptor_out(h_pitch)

        if skip_only:
            # Train only the skip adaptor
            h_pitch = h_pitch.detach()
            notes_logits = notes_logits.detach()

        # Merged branch
        x = h_pitch + h_skip
        x = self.wav_lm.gpt.forward_after(overwrite_hidden_states=x, inputs_embeds=x).last_hidden_state
        
        # Decode
        x = x.reshape(batch_size * seq_len, -1, 1, 1) # [batch_size * seq_len, emb_size, 1, 1]
        x = self.wav_lm.decoder(x)
        x = x.reshape(batch_size, seq_len, x.shape[2], x.shape[3]) # [batch_size, seq_len, 128, 16]
        return x, notes_logits