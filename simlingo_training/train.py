import os
import hydra

from omegaconf import OmegaConf
import torch
import wandb

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from transformers import AutoProcessor

from simlingo_training.utils.logging_project import setup_logging, sync_wandb

from simlingo_training.config import TrainConfig
from simlingo_training.callbacks.visualise import VisualiseCallback


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)
    
    # TEST MODE: Set to True for quick testing with limited batches
    # Set to False for full training run
    TEST_MODE = True  # <--- CHANGE THIS TO False FOR FULL RUN; True for testing
    TEST_NUM_BATCHES = 10000  # Number of batches for testing
    
    # CRASH MODE: Set to True to finetune on ambiguous crash dataset
    FINETUNE_CRASH_MODE = False  # <--- Set to True for crash finetuning
    
    # Crash finetuning configuration
    CRASH_CHECKPOINT = '/home/lijack/Documents/cs230-final-project/outputs/simlingo/checkpoints/epoch=013.ckpt'
    CRASH_MAX_EPOCHS = 3  # Number of finetuning epochs (starts from 0 since we only load weights, not full checkpoint)
    CRASH_LEARNING_RATE = 1e-5  # Lower LR for finetuning (original: 3e-5)

    # turn off wandb uploading when in debug mode or test mode
    if cfg.debug or TEST_MODE:
        os.environ["WANDB_MODE"] = "offline"
    
    # Add suffix to wandb name for crash mode
    if FINETUNE_CRASH_MODE:
        cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}_crash_finetune"
    else:
        cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"
    
    if TEST_MODE:
        print("="*60)
        print(f"⚠️  TEST MODE ENABLED - Training on {TEST_NUM_BATCHES} batches only")
        print("   Set TEST_MODE=False in train.py for full training")
        print("="*60)
    
    # Configure for crash mode finetuning if enabled
    if FINETUNE_CRASH_MODE:
        print("="*60)
        print("🔥 CRASH MODE FINETUNING ENABLED")
        print(f"   Loading checkpoint weights: {CRASH_CHECKPOINT}")
        print("   Using ambiguous crash training dataset")
        print(f"   Training epochs: {CRASH_MAX_EPOCHS} (starting from 0)")
        print(f"   Learning rate: {CRASH_LEARNING_RATE} (reduced for finetuning)")
        print("="*60)
        
        # Use ambiguous_crash folder for dreamer data
        cfg.data_module.base_dataset.dreamer_folder = "ambiguous_crash"
        
        # Disable crash mode filtering (ambiguous_crash folder only contains crash samples)
        cfg.data_module.base_dataset.filter_dreamer_mode = None
        
        # Train ONLY on crash dreamer data (disable other datasets)
        cfg.data_module.driving_dataset = None
        cfg.data_module.qa_dataset = None
        # Keep dreamer_dataset active (this is the crash data)
        
        # Set checkpoint to load
        if cfg.checkpoint is None:
            cfg.checkpoint = CRASH_CHECKPOINT
        print(f"Checkpoint: {cfg.checkpoint}")
        
        # Adjust training parameters for finetuning
        cfg.max_epochs = CRASH_MAX_EPOCHS
        cfg.model.lr = CRASH_LEARNING_RATE
        
        # Enable safety flag for crash training
        cfg.data_module.base_dataset.use_safety_flag = True
        
        # Freeze vision encoder for efficient finetuning (optional - set to False to train vision encoder)
        FREEZE_VISION_ENCODER = False  # Recommended for crash finetuning
        cfg.model.vision_model.freeze = FREEZE_VISION_ENCODER
        
        print(f"Dreamer folder: {cfg.data_module.base_dataset.dreamer_folder}")
        print(f"Filter dreamer mode: {cfg.data_module.base_dataset.filter_dreamer_mode}")
        print(f"Training: {CRASH_MAX_EPOCHS} epochs (epoch 0 to {CRASH_MAX_EPOCHS-1})")
        print(f"Datasets: dreamer ONLY (driving/qa disabled)")
        print(f"Safety flag: {cfg.data_module.base_dataset.use_safety_flag}")
        print(f"Vision encoder frozen: {FREEZE_VISION_ENCODER} (308M params)")
        print(f"Language model: LoRA enabled (only 17.6M trainable params)")
    
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = None #f"pretrained/{(model_type_name)}"
    
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False
    )
    
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        # Use strict=False to allow missing keys for new heads (contrastive_head, safety_gate)
        print("Loading state dict with strict=False to allow new heads initialization")
        model.load_state_dict(state_dict, strict=False)

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"
    
    if cfg.overfit > 0:
        overfit = cfg.overfit
        
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = cfg.resume_path
    resume_wandb = False
    
    # If starting from a specific checkpoint but not resuming training state (finetuning)
    if cfg.checkpoint is not None and not cfg.resume:
        print(f"Loading weights from {cfg.checkpoint} for finetuning (strict=False)")
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        
        # Load weights with strict=False to allow missing keys (new heads)
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")

    # if folder for this experiment does not exist set resume to true
    # to create necessary folders to resume wandb logging later
    if resume_path is not None and not os.path.exists(resume_path):
        resume_wandb = True
    elif resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True

    if resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None

    # setup lightning logger
    loggers = []
    # csvlogger = CSVLogger("log/", "CSVLogger")
    # loggers.append(csvlogger)
    # csvlogger = None

    wandblogger = WandbLogger(
        project=cfg.wandb_project,
        id=cfg.wandb_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume=resume_wandb,
    )
    wandblogger.watch(model)
    loggers.append(wandblogger)

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        monitor=None,
        dirpath="./checkpoints",
        filename="{epoch:03d}",
        save_last=True,
        every_n_epochs=cfg.val_every_n_epochs,
        # every_n_train_steps=cfg.val_check_interval,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=3)
    callbacks=[
        checkpoint_callback, 
        model_summary, 
        # ThroughputMonitor(batch_size_fn=lambda batch: batch.driving_input.camera_images.size(0)), 
        VisualiseCallback(interval=1000, val_interval=1000)
    ]
    if not cfg.debug: 
        callbacks.append(lr_monitor)
    
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    # Configure batch limits for TEST_MODE
    limit_train_batches = TEST_NUM_BATCHES if TEST_MODE else None
    limit_val_batches = TEST_NUM_BATCHES if TEST_MODE else None
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            devices=cfg.gpus,
            # enable_checkpointing=False,
            gradient_clip_val=0.3,
            # gradient_clip_algorithm="value",
            # log_every_n_steps=10,
            logger=loggers,
            # max_steps=cfg.max_steps,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            # use_distributed_sampler=False,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
            # val_check_interval=cfg.val_check_interval,
            limit_train_batches=limit_train_batches,  # NEW: Limit batches for testing
            limit_val_batches=limit_val_batches,  # NEW: Limit val batches for testing
        )

    trainer.fit(model, data_module, ckpt_path=resume_path)
    wandb.finish()

if __name__ == "__main__":
    main()