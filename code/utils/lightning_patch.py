import lightning

def skip_loading_optimizers():
    lightning.pytorch.trainer.connectors.checkpoint_connector._CheckpointConnector.restore_optimizers_and_schedulers = lambda *args, **kwargs: None
    return