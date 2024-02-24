import os
import datetime
import argparse
import tempfile

import torch
import lightning

import utils.globals as uglobals
import utils.logging_utils as logging_utils
from utils.wav_dataset import make_wav_loader

from models.spectogram_rvqvae import Spectorgram_RVQVAE
from models.audio_lm import AudioLM
from models.cascading_audio_lm import CascadingAudioLM
from models.deterministic_cheeseburger import DeterministicCheeseburger
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
    logger_dir = f'{uglobals.RUNS_DIR}/{args.task}'
    logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir = logger_dir, 
        name=args.name, 
        version=f'{args.mode}_{date_str}',
    )
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=f'{logger_dir}/{args.name}/checkpoints',
        save_last=True,
        save_top_k=1,
        monitor='val/monitor'
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
    
    # Trainer
    trainer = lightning.Trainer(
        max_epochs=args.max_n_epochs, 
        check_val_every_n_epoch=args.eval_n_epoch,
        accelerator=accelerator,
        logger=logger,
        deterministic=not args.nondeterministic,
        num_sanity_val_steps=2,
        enable_progress_bar=args.single_worker,
        log_every_n_steps=len(train_loader)//5 if not args.debug else 1, # Log 5 times per epoch
        callbacks=[checkpoint_callback],
        inference_mode=False if (args.task in['spectrogram_rvqvae', 'det_cheeseburger', 'audio_lm', 'det_wav_tf'] and args.mode=='predict_dev') else True, # Enable grad for reverse mel spectrogram transforms
        limit_train_batches=3 if args.debug else 1.0,
        limit_val_batches=3 if args.debug else 1.0,
        # limit_test_batches=3 if args.debug else 1.0,
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
        trainer.predict(model, dataloaders=dev_loader, ckpt_path=args.checkpoint, return_predictions=False)
    else:
        raise NotImplementedError
    
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basics
    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--force_restart_training', action='store_true') # Otherwise, automatically resume the last checkpoint
    parser.add_argument('--nondeterministic', action='store_true')
    parser.add_argument('--single_worker', action='store_true')
    
    # Formulation
    parser.add_argument('--task', type=str, default=None, choices=['spectrogram_rvqvae', 'audio_lm', 'cascade_audio_lm', 'det_cheeseburger', 'det_wav_tf', 'pitch_lm'])
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

    # Training: Pitch_LM
    parser.add_argument('--pitch_lm_config', type=str, default='distilgpt2', choices=['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large'])

    # Prediction
    parser.add_argument('--n_prediction_batches', default=3, type=int)
    parser.add_argument('--test_context_len', default=3, type=int)

    args = parser.parse_args()
    args.uglobals = logging_utils.module_to_dict(uglobals)

    if args.debug:
        args.name = 'debug'
        args.single_worker = True

        args.task = 'det_cheeseburger'
        args.mode = 'train'

        args.training_mode = 'skip_branch'
        
        args.batch_size = 3
        args.max_n_epochs = 4

    main(args)