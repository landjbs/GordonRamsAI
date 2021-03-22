class Trainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config: DictConfig):
        # filing
        if config.file.log:
            # build logger
            logger = TensorBoardLogger(
                save_dir=config.file.save_path,
                name=config.file.run_name,
                # log_graph=config. "type is long run"
            )
            # configure checkpoints
            save_dir = (
                f'{config.file.save_path}/{config.file.run_name}'
                f'/version_{logger.version}'
            )
            checkpoint_callback = ModelCheckpoint(
                filepath=(
                    f'{save_dir}/checkpoints'
                ),
                # save_last=True
                # monitor='epoch',
                # save_top_k=1,
                # mode='max',
            )
            callbacks = [checkpoint_callback]
            print(f'\n\nView logs at:\ntensorboard --logdir={save_dir}\n\n')
        else:
            logger = None
            callbacks = None
        return cls(
            # training
            deterministic=True,
            # gradients
            gradient_clip_val=(
                config.optimizer.grad_clip if config.optimizer.grad_clip else 0
            ),
            # compute
            auto_select_gpus=(
                True if (config.hardware in ('auto', 'gpu')) else False
            ),
            precision=(
                config.compute.precision if (config.compute.hardware != 'cpu')
                else 32
            ),
            # logging
            logger=logger,
            # profiler=config.profiler,
            log_every_n_steps=config.steps.log_every,
            val_check_interval=config.steps.val_every,
            callbacks=callbacks,
            resume_from_checkpoint=config.file.load_path
        )
