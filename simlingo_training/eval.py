import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoProcessor, AutoTokenizer

from simlingo_training.config import TrainConfig
from simlingo_training.utils.logging_project import setup_logging
from simlingo_training.callbacks import IncrementalPredictionWriter
# from simlingo_training.callbacks.visualise import VisualiseCallback

@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42)
    
    # TEST MODE: Set to True for quick testing with 10 samples
    # Set to False for full evaluation run
    TEST_MODE = False  # <--- CHANGE THIS TO False FOR FULL RUN; True for testing
    
    # eval_mode = "QA"
    # eval_mode = "commentary"
    eval_mode = "Dreaming"
    #     eval_mode = "crash"  # Uses ambiguous crash dataset with same eval as Dreaming

    qa_dataset = cfg.data_module.qa_dataset
    insteval_dataset = cfg.data_module.insteval_dataset
    
    # Prioritize checkpoint from CLI if provided
    load_path = cfg.checkpoint
    if load_path is None:
        load_path = '/home/lijack/Documents/cs230-final-project/outputs/simlingo/checkpoints/epoch=013.ckpt'
        
    if load_path is not None:
        # Load config associated with the checkpoint
        load_path_config = Path(load_path).parent.parent / '.hydra/config.yaml'
        if load_path_config.exists():
            print(f"Loading config from: {load_path_config}")
            ckpt_cfg = OmegaConf.load(load_path_config)
            cfg = ckpt_cfg
        else:
             print(f"Warning: Config not found at {load_path_config}. Using current config.")

    # Ensure checkpoint path is set correctly in cfg
    if load_path is not None:
        cfg.checkpoint = str(load_path)

    cfg.data_module.qa_dataset = qa_dataset
    cfg.data_module.insteval_dataset = insteval_dataset
    cfg.gpus = 1
    cfg.data_module.num_workers = 4
    cfg.data_module.batch_size = 128

    if TEST_MODE:
        print("="*60)
        print("⚠️  TEST MODE ENABLED - Running on 30 samples only")
        print("   Set TEST_MODE=False in eval.py for full evaluation")
        print("="*60)
    
    print(f'Eval mode: {eval_mode}')
    print(f'Checkpoint: {load_path}')
    print(f"Using {cfg.gpus} GPUs")
    print(f'Batch size: {cfg.data_module.batch_size}')
    
    if eval_mode == "QA" or eval_mode == "commentary":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.insteval_dataset = None 
    elif eval_mode == "Dreaming" or eval_mode == "crash":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.qa_dataset = None
        
        # For crash mode, use ambiguous crash subfolder within same dataset
        if eval_mode == "crash":
            cfg.data_module.base_dataset.dreamer_folder = "ambiguous_crash"
            print(f"Using crash dreamer folder: {cfg.data_module.base_dataset.dreamer_folder}")
    
    if eval_mode == "QA":
        cfg.data_module.base_dataset.use_commentary = False
        cfg.data_module.base_dataset.use_qa = True
    elif eval_mode == "commentary":
        cfg.data_module.base_dataset.use_commentary = True
        cfg.data_module.base_dataset.use_qa = False
    elif eval_mode == "Dreaming" or eval_mode == "crash":
        # cfg.data_module.base_dataset.use_safety_flag = True
        cfg.data_module.base_dataset.use_safety_flag = False

    print(f'Use safety flag: {cfg.data_module.base_dataset.use_safety_flag}')
    
    
    # disable image augmentation
    cfg.data_module.base_dataset.img_augmentation = False
    
    # disable img_shift_augmentation
    cfg.data_module.base_dataset.img_shift_augmentation = False
    
    if "2B" in cfg.model.language_model.variant:
        processor = AutoTokenizer.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = f"pretrained/{(model_type_name)}"
    
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=True,
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
        model.load_state_dict(state_dict)

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"

    
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = "./checkpoints/last.ckpt"


    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None
    
    # setup lightning logger
    loggers = []

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )
  
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    # Add prediction writer callback to save predictions incrementally
    prediction_writer = IncrementalPredictionWriter()
    
    # Configure prediction limit based on TEST_MODE
    # With batch_size=10 and limit=3, we get exactly 30 samples for testing
    limit_batches = 3 if TEST_MODE else None
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            devices=cfg.gpus,
            gradient_clip_val=0.3,
            log_every_n_steps=20,
            logger=loggers,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
            callbacks=[prediction_writer],
            limit_predict_batches=limit_batches,  # NEW: Limit batches for testing
        )

    if load_path is not None:
        trainer.predict(model, data_module, ckpt_path=f"{load_path}/")
    else:
        trainer.predict(model, data_module)

if __name__ == "__main__":
    main()