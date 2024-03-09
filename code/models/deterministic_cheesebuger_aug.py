import itertools
from copy import deepcopy

import torch

from models.deterministic_cheeseburger import DeterministicCheeseburger

class DeterministicCheeseburgerAugZ(DeterministicCheeseburger):
    def __init__(self, args, sr):
        super().__init__(args, sr)

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
        return x, notes_logits, h_skip, h_pitch
    
    def joint_loss_and_log(self, spectrogram_pred, spectrogram_target, notes_logits, notes_target, h_skip, h_pitch, batch_size, log_name, skip_only=False):

        mse = torch.nn.MSELoss()(spectrogram_pred, spectrogram_target)
        ce = torch.nn.CrossEntropyLoss()(notes_logits.reshape(-1, notes_logits.shape[-1]), notes_target.reshape(-1))

        # Z augmentation losses
        mse_timbre = torch.nn.MSELoss()(h_skip[2::3], h_skip[0::3])
        mse_pitch = torch.nn.MSELoss()(h_pitch[2::3], h_pitch[1::3])

        loss = mse + self.args['det_cheese_ce_weight'] * ce + self.args['det_cheese_aug_z_timbre_weight'] * mse_timbre + self.args['det_cheese_aug_z_pitch_weight'] * mse_pitch
        accuracy = (notes_logits.argmax(-1) == notes_target).float().mean()

        self.log(f'{log_name}/loss', loss, batch_size=batch_size)
        self.log(f'{log_name}/monitor', loss, batch_size=batch_size) # Keep the best checkpoint based on this metric
        self.log(f'{log_name}/mse_out', mse, batch_size=batch_size)
        self.log(f'{log_name}/ce', ce, batch_size=batch_size)
        self.log(f'{log_name}/mse_timbre', mse_timbre, batch_size=batch_size)
        self.log(f'{log_name}/mse_pitch', mse_pitch, batch_size=batch_size)
        self.log(f'{log_name}/accuracy', accuracy, batch_size=batch_size)

        training_mode = 0
        if skip_only:
            training_mode = 3
        self.log(f'{log_name}/training_mode', training_mode, batch_size=batch_size)
        return loss
    
    def batch_to_loss(self, batch, name):
        wav, spectrogram_in, spectrogram_target, notes_in, notes_target, batch_size, seq_len = self.prep_batch(batch)

        if self.training_mode == 'joint':
            spectrogram_pred, notes_logits, h_skip, h_pitch = self.joint_forward(spectrogram_in)
            loss = self.joint_loss_and_log(spectrogram_pred, spectrogram_target, notes_logits, notes_target, h_skip, h_pitch, batch_size, name)
        else:
            raise NotImplementedError
        return loss
    
    def infer(self, spectrogram_in, notes_in):
        seq_len = spectrogram_in.shape[1] + 1
        spectrogram_in = spectrogram_in[:, :self.test_context_len]
        notes_in = notes_in[:, 1: 1 + self.test_context_len] # Remove the BoS token
        
        while spectrogram_in.shape[1] < seq_len:
            spectrogram_pred, notes_logits, _, _ = self.joint_forward(deepcopy(spectrogram_in.detach()))
            notes_pred = notes_logits.argmax(-1)
            spectrogram_in = torch.cat([spectrogram_in, spectrogram_pred[:, -1:]], dim=1).detach()
            notes_in = torch.cat([notes_in, notes_pred[:, -1:]], dim=1).detach()
            
        return spectrogram_in, notes_in
    

class DeterministicCheeseburgerAugX(DeterministicCheeseburger):
    def __init__(self, args, sr):
        super().__init__(args, sr)

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
        x_recombined = h_pitch[1::3] + h_skip[::3]
        x = torch.cat([x, x_recombined], dim=0)
        x = self.wav_lm.gpt.forward_after(overwrite_hidden_states=x, inputs_embeds=x).last_hidden_state
        
        # Decode
        x = x.reshape(batch_size * seq_len * 4 // 3, -1, 1, 1) # [batch_size * seq_len, emb_size, 1, 1]
        x = self.wav_lm.decoder(x)
        x = x.reshape(-1, seq_len, x.shape[2], x.shape[3]) # [batch_size, seq_len, 128, 16]
        return x, notes_logits
    
    def joint_loss_and_log(self, spectrogram_pred, spectrogram_target, notes_logits, notes_target, batch_size, log_name, skip_only=False):

        mse = torch.nn.MSELoss()(spectrogram_pred[:batch_size], spectrogram_target)
        mse_aug = torch.nn.MSELoss()(spectrogram_pred[batch_size:], spectrogram_target[2::3])
        ce = torch.nn.CrossEntropyLoss()(notes_logits.reshape(-1, notes_logits.shape[-1]), notes_target.reshape(-1))

        loss = mse + self.args['det_cheese_ce_weight'] * ce + self.args['det_cheese_aug_x_weight'] * mse_aug
        accuracy = (notes_logits.argmax(-1) == notes_target).float().mean()

        self.log(f'{log_name}/loss', loss, batch_size=batch_size)
        self.log(f'{log_name}/monitor', loss, batch_size=batch_size) # Keep the best checkpoint based on this metric
        self.log(f'{log_name}/mse_out', mse, batch_size=batch_size)
        self.log(f'{log_name}/ce', ce, batch_size=batch_size)
        self.log(f'{log_name}/mse_aug', mse_aug, batch_size=batch_size)
        self.log(f'{log_name}/accuracy', accuracy, batch_size=batch_size)

        training_mode = 0
        if skip_only:
            training_mode = 3
        self.log(f'{log_name}/training_mode', training_mode, batch_size=batch_size)
        return loss
    
    def infer(self, spectrogram_in, notes_in):
        seq_len = spectrogram_in.shape[1] + 1
        spectrogram_in = spectrogram_in[:, :self.test_context_len]
        notes_in = notes_in[:, 1: 1 + self.test_context_len] # Remove the BoS token
        
        while spectrogram_in.shape[1] < seq_len:
            spectrogram_pred, notes_logits, _, _ = self.joint_forward_plain(deepcopy(spectrogram_in.detach()))
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