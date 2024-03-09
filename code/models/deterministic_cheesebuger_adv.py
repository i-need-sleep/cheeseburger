import itertools
from copy import deepcopy

import torch

from models.deterministic_cheeseburger import DeterministicCheeseburger
from models.modeling_utils.adaptor import Adaptor
from models.modeling_utils.grad_reverse import grad_reverse

class DeterministicCheeseburgerAdv(DeterministicCheeseburger):
    def __init__(self, args, sr):
        super().__init__(args, sr)
        self.pitch_adversary = Adaptor(self.wav_lm.gpt_config.n_embd, self.pitch_lm.gpt_config.vocab_size)

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

        # Adversarial branch
        h_adv = grad_reverse(h_skip)
        adv_logits = self.pitch_adversary(h_adv) # TODO

        # Pitch branch
        x = self.adaptor_in(x)
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
        return x, notes_logits, adv_logits
    
    def joint_loss_and_log(self, spectrogram_pred, spectrogram_target, notes_logits, adv_logits, notes_target, batch_size, log_name, skip_only=False):
        mse = torch.nn.MSELoss()(spectrogram_pred, spectrogram_target)
        ce = torch.nn.CrossEntropyLoss()(notes_logits.reshape(-1, notes_logits.shape[-1]), notes_target.reshape(-1))
        ce_adv = torch.nn.CrossEntropyLoss()(adv_logits.reshape(-1, adv_logits.shape[-1]), notes_target.reshape(-1))
        loss = mse + self.args['det_cheese_ce_weight'] * ce + self.args['det_cheese_adv_weight'] * ce_adv
        accuracy = (notes_logits.argmax(-1) == notes_target).float().mean()
        accuracy_adv = (adv_logits.argmax(-1) == notes_target).float().mean()

        self.log(f'{log_name}/loss', loss, batch_size=batch_size)
        self.log(f'{log_name}/monitor', mse, batch_size=batch_size) # Keep the best checkpoint based on this metric
        self.log(f'{log_name}/mse', mse, batch_size=batch_size)
        self.log(f'{log_name}/ce', ce, batch_size=batch_size)
        self.log(f'{log_name}/ce_adv', ce_adv, batch_size=batch_size)
        self.log(f'{log_name}/accuracy', accuracy, batch_size=batch_size)
        self.log(f'{log_name}/accuracy_adv', accuracy_adv, batch_size=batch_size)

        training_mode = 0
        if skip_only:
            training_mode = 3
        self.log(f'{log_name}/training_mode', training_mode, batch_size=batch_size)
        return loss
    
    def batch_to_loss(self, batch, name):
        wav, spectrogram_in, spectrogram_target, notes_in, notes_target, batch_size, seq_len = self.prep_batch(batch)

        if self.training_mode == 'joint':
            spectrogram_pred, notes_logits, adv_logits = self.joint_forward(spectrogram_in)
            loss = self.joint_loss_and_log(spectrogram_pred, spectrogram_target, notes_logits, adv_logits, notes_target, batch_size, name)
        elif self.training_mode == 'pre_branch':
            notes_logits = self.pre_branch_forward(spectrogram_in)
            loss = self.pre_branch_loss_and_log(notes_logits, notes_target, batch_size, name)
        elif self.training_mode == 'post_branch':
            spectrogram_pred = self.post_branch_forward(notes_in)
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
            spectrogram_pred, notes_logits, adv_logits = self.joint_forward(deepcopy(spectrogram_in.detach()))
            notes_pred = notes_logits.argmax(-1)
            spectrogram_in = torch.cat([spectrogram_in, spectrogram_pred[:, -1:]], dim=1).detach()
            notes_in = torch.cat([notes_in, notes_pred[:, -1:]], dim=1).detach()
            
        return spectrogram_in, notes_in