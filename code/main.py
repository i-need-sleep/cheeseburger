import os
import datetime
import argparse
import tempfile

import torch
import lightning

import utils.globals as uglobals
import utils.logging_utils as logging_utils
import utils.lightning_patch as lightning_patch
from utils.wav_dataset import make_wav_loader, make_augmented_wav_loader

from models.spectogram_rvqvae import Spectorgram_RVQVAE
from models.audio_lm import AudioLM
from models.cascading_audio_lm import CascadingAudioLM
from models.deterministic_cheeseburger import DeterministicCheeseburger
from models.deterministic_cheesebuger_adv import DeterministicCheeseburgerAdv
from models.deterministic_cheesebuger_aug import DeterministicCheeseburgerAugZ, DeterministicCheeseburgerAugX
from models.deterministic_cheesebuger_unsup import DeterministicCheeseburgerUnsupervised
from models.deterministic_wav_transformer import DeterministicWavTransformer
from models.pitch_lm import PitchLM

def main(args):
    # The default temp dir does not work on the cluster
    tempfile.tempdir = uglobals.TEMP_DIR

    # Seeding
    if not args.nondeterministic:
        lightning.seed_everything(uglobals.SEED)

    # Device
    if not torch.cuda.is_available() or args.force_cpu:
        device = torch.device('cpu')
        accelerator = 'cpu'
    else:
        device = torch.device('cuda')
        accelerator = 'gpu'
        torch.set_float32_matmul_precision('high')
    print(f'Device: {device}')

    # Logging and checkpointing
    date_str = str(datetime.datetime.now())[:-7].replace(':','-').replace(' ', '_')
    logger_dir = f'{uglobals.RUNS_DIR}/{args.task}/{args.experiment_group}'
    logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir = logger_dir, 
        name=args.name, 
        version=f'{args.mode}_{date_str}',
    )
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=f'{logger_dir}/{args.name}/checkpoints',
        save_last=True,
        save_top_k=1,
        monitor='val/monitor' # Minimized
    )

    # Resume from the last checkpoint
    if not args.force_restart_training and args.checkpoint is None:
        last_checkpoint = f'{logger_dir}/{args.name}/checkpoints/last.ckpt'
        if os.path.exists(last_checkpoint):
            args.checkpoint = last_checkpoint
            print(f'Resuming from the last checkpoint: {args.checkpoint}')
            
    # Print and save args
    logging_utils.print_and_save_args_uglobals(args, logger)
    
    # Create model and data loaders
    # This should be the only place to change when we add new tasks/models
    if args.task == 'spectrogram_rvqvae':
        sr = 16000
        train_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        test_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        model = Spectorgram_RVQVAE(vars(args), sr=sr)
    elif args.task == 'audio_lm':
        sr = 16000
        train_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=args.mode=='predict_dev', single_worker=args.single_worker)
        test_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        model = AudioLM(vars(args))
    elif args.task == 'cascade_audio_lm':
        sr = 16000
        train_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        test_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        model = CascadingAudioLM(vars(args))    
    elif args.task == 'det_cheeseburger':
        sr = 16000
        train_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=args.mode=='predict_dev', single_worker=args.single_worker)
        test_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        model = DeterministicCheeseburger(vars(args), sr)
    elif args.task == 'det_cheeseburger_adv':
        sr = 16000
        train_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=args.mode=='predict_dev', single_worker=args.single_worker)
        test_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        model = DeterministicCheeseburgerAdv(vars(args), sr)
    elif args.task == 'det_cheeseburger_aug_z':
        sr = 16000
        train_loader = make_augmented_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_FLAT_VELO_WAV_DIR, args.batch_size//3, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_augmented_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_FLAT_VELO_WAV_DIR, args.batch_size//3, sr, shuffle=args.mode=='predict_dev', single_worker=args.single_worker)
        test_loader = make_augmented_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_FLAT_VELO_WAV_DIR, args.batch_size//3, sr, shuffle=False, single_worker=args.single_worker)
        model = DeterministicCheeseburgerAugZ(vars(args), sr)
    elif args.task == 'det_cheeseburger_aug_x':
        sr = 16000
        train_loader = make_augmented_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_FLAT_VELO_WAV_DIR, args.batch_size//3, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_augmented_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_FLAT_VELO_WAV_DIR, args.batch_size//3, sr, shuffle=args.mode=='predict_dev', single_worker=args.single_worker)
        test_loader = make_augmented_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_FLAT_VELO_WAV_DIR, args.batch_size//3, sr, shuffle=False, single_worker=args.single_worker)
        model = DeterministicCheeseburgerAugX(vars(args), sr)
    elif args.task == 'det_cheeseburger_unsup':
        sr = 16000
        train_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=args.mode=='predict_dev', single_worker=args.single_worker)
        test_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        model = DeterministicCheeseburgerUnsupervised(vars(args), sr)
    elif args.task == 'det_wav_tf':
        sr = 16000
        train_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=args.mode=='predict_dev', single_worker=args.single_worker)
        test_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        model = DeterministicWavTransformer(vars(args), sr=sr)
    elif args.task == 'pitch_lm':
        sr = 16000
        train_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/train_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/dev_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=args.mode=='predict_dev', single_worker=args.single_worker)
        test_loader = make_wav_loader(f'{uglobals.TOY_16K_TRAINING_DIR}/test_midi.pt', uglobals.TOY_16K_WAV_DIR, args.batch_size, sr, shuffle=False, single_worker=args.single_worker)
        model = PitchLM(vars(args))
    else:
        raise NotImplementedError
    
    # Overwriting checkpoint loading
    if args.no_stict_loading:
        model.strict_loading = False
    if args.reinit_optimizers:
        lightning_patch.skip_loading_optimizers()

    # Trainer
    trainer = lightning.Trainer(
        max_epochs=args.max_n_epochs, 
        check_val_every_n_epoch=args.eval_n_epoch,
        accelerator=accelerator,
        logger=logger,
        deterministic=not args.nondeterministic,
        num_sanity_val_steps=1,
        enable_progress_bar=args.single_worker,
        log_every_n_steps=len(train_loader)//5 if not args.debug else 1, # Log 5 times per epoch
        callbacks=[checkpoint_callback],
        # inference_mode=False if (args.task in['spectrogram_rvqvae', 'det_cheeseburger', 'audio_lm', 'det_wav_tf'] and args.mode=='predict_dev') else True, # Enable grad for reverse mel spectrogram transforms
        inference_mode=False if args.mode=='predict_dev' else True, # Enable grad for reverse mel spectrogram transforms
        limit_train_batches=3 if args.debug else 1.0,
        limit_val_batches=3 if args.debug else 1.0,
        limit_test_batches=3 if args.debug else 1.0,
        limit_predict_batches= args.n_prediction_batches,
    )

    if args.mode == 'train':
        model.training_mode = args.training_mode
        trainer.fit(model, train_loader, dev_loader, ckpt_path=args.checkpoint)
    elif args.mode == 'test':
        model.test_context_len = args.test_context_len
        trainer.test(model, dataloaders=test_loader, ckpt_path=args.checkpoint)
    elif args.mode == 'test_dev':
        model.test_context_len = args.test_context_len
        trainer.test(model, dataloaders=dev_loader, ckpt_path=args.checkpoint)
    elif args.mode == 'predict_dev':
        model.test_context_len = args.test_context_len
        model.intervention_mode = args.intervention_mode
        model.intervention_step = args.intervention_step
        trainer.predict(model, dataloaders=dev_loader, ckpt_path=args.checkpoint, return_predictions=False)
    else:
        raise NotImplementedError
    
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Names
    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--experiment_group', type=str, default='unnamed')

    # Checkpointing
    parser.add_argument('--force_restart_training', action='store_true') # Otherwise, automatically resume the last checkpoint
    parser.add_argument('--no_stict_loading', action='store_true')
    parser.add_argument('--reinit_optimizers', action='store_true')

    # Debugging
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--single_worker', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--nondeterministic', action='store_true')

    # Formulation
    parser.add_argument('--task', type=str, default=None, choices=['spectrogram_rvqvae', 'audio_lm', 'cascade_audio_lm', 'det_cheeseburger', 'det_cheeseburger_adv', 'det_cheeseburger_aug_x', 'det_cheeseburger_aug_z', 'det_cheeseburger_unsup', 'det_wav_tf', 'pitch_lm'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_dev', 'predict_dev'])

    # Training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_n_epochs', default=-1, type=int)
    parser.add_argument('--eval_n_epoch', default=1, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--lr_scheduler_start_factor', default=1, type=float)
    parser.add_argument('--lr_scheduler_warmup_epochs', default=1, type=int)
    parser.add_argument('--lr_scheduler_end_factor', default=1, type=float)
    parser.add_argument('--lr_scheduler_anneal_epochs', default=1, type=int)
    
    # Different training step functions
    parser.add_argument('--training_mode', default=None, type=str)

    # Training: Spectrogram_RVQVAE
    parser.add_argument('--commit_loss_weight', default=1, type=float)
    parser.add_argument('--n_quantizers', default=8, type=int)

    # Training: Audio_LM
    parser.add_argument('--lm_config', type=str, default='distilgpt2', choices=['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument('--rvqvae_checkpoint', default='', type=str)

    # Training: Cascading_Audio_LM
    parser.add_argument('--downstream_lm_config', type=str, default='distilgpt2', choices=['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large'])

    # Training: Deterministic Cheeseburger
    parser.add_argument('--det_cheese_wav_lm_checkpoint', default='../pretrained/deterministic/det_wav_lm.ckpt', type=str)
    parser.add_argument('--det_cheese_pitch_lm_checkpoint', default='../pretrained/deterministic/pitch_lm.ckpt', type=str)
    parser.add_argument('--det_cheese_insertion_layer', default=5, type=int)
    parser.add_argument('--det_cheese_ce_weight', default=1, type=float)

    # Training: Deterministic Cheeseburger variants
    parser.add_argument('--det_cheese_softmax_logits', action='store_true')
    parser.add_argument('--det_cheese_adv_weight', default=1, type=float)

    parser.add_argument('--det_cheese_aug_z_timbre_weight', default=1, type=float)
    parser.add_argument('--det_cheese_aug_z_pitch_weight', default=1, type=float)
    parser.add_argument('--det_cheese_aug_x_weight', default=1, type=float)
    parser.add_argument('--det_cheese_aug_x_no_pitch_model', action='store_true')

    parser.add_argument('--det_cheese_unsup_adv_weight', default=1, type=float)
    parser.add_argument('--det_cheese_unsup_std_weight', default=1, type=float)
    parser.add_argument('--det_cheese_unsup_cov_weight', default=1, type=float)
    parser.add_argument('--det_cheese_unsup_cov_margin', default=10, type=float)

    # Training: Pitch_LM
    parser.add_argument('--pitch_lm_config', type=str, default='distilgpt2', choices=['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large'])

    # Prediction
    parser.add_argument('--n_prediction_batches', default=4, type=int)
    parser.add_argument('--test_context_len', default=4, type=int)
    
    # Prediction: Intervention
    parser.add_argument('--intervention_mode', default='', type=str) # swap, 01, sample_patch
    parser.add_argument('--intervention_step', default='', type=str)
    
    args = parser.parse_args()
    args.uglobals = logging_utils.module_to_dict(uglobals)

    if args.debug:
        args.name = 'debug'
        args.experiment_group = 'debug'
        args.single_worker = True

        args.task = 'det_cheeseburger_unsup'
        args.mode = 'train'
        
        args.training_mode = 'joint'
        # args.checkpoint = '../results/runs/det_cheeseburger/finetuned_softmax_3e-4.ckpt'
        # args.no_stict_loading = True
        # args.reinit_optimizers = True
        
        args.batch_size = 6
        args.max_n_epochs = 40

        # for name in ['skip', 'finetuned', 'joint']:
        #     for mode in ['', 'sample_patch', 'swap', '+-1e6']:
        #         for step in ['last', 'all']:
        #             if step == 'last' and mode in ['sample_patch', '']:
        #                 continue
        #             args.intervention_mode = mode
        #             args.intervention_step = step
        #             args.checkpoint = f'../results/runs/det_cheeseburger/{name}_softmax_3e-4.ckpt'
        #             args.name = f'{name}_{mode}_{step}'
        #             main(args)
        # exit()

        # args.intervention_mode = 'sample_patch'
        # args.intervention_step = 'all'
        # args.checkpoint = f'../results/runs/det_cheeseburger/unsup_poc.ckpt'

    main(args)