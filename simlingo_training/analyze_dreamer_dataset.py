import os
from pathlib import Path
from collections import Counter, defaultdict

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm

from simlingo_training.config import TrainConfig


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42)
    
    # Set eval mode to Dreaming (same as eval.py)
    eval_mode = "Dreaming"
    
    # Load checkpoint path config if specified
    load_path = 'outputs/simlingo/checkpoints/epoch=013.ckpt'
    if load_path is not None:
        load_path_config = Path(load_path).parent.parent / '.hydra/config.yaml'
        if load_path_config.exists():
            cfg = OmegaConf.load(load_path_config)
    
    # Keep the insteval_dataset for dreamer evaluation
    insteval_dataset = cfg.data_module.insteval_dataset
    
    # Set GPUs and workers
    cfg.gpus = 1
    cfg.data_module.num_workers = 8
    cfg.data_module.batch_size = 64
    
    print(f'Eval mode: {eval_mode}')
    print(f"Using {cfg.gpus} GPUs")
    
    # Configure for Dreaming mode (same as eval.py lines 47-59)
    cfg.data_module.dreamer_dataset = None
    cfg.data_module.driving_dataset = None
    cfg.data_module.qa_dataset = None
    cfg.data_module.insteval_dataset = insteval_dataset
    
    cfg.data_module.base_dataset.use_safety_flag = True
    
    # Disable image augmentation
    cfg.data_module.base_dataset.img_augmentation = False
    cfg.data_module.base_dataset.img_shift_augmentation = False
    
    # Load processor (same as eval.py)
    if "2B" in cfg.model.language_model.variant:
        processor = AutoTokenizer.from_pretrained(
            cfg.model.language_model.variant, 
            trust_remote_code=True, 
            use_fast=False
        )
    else:
        processor = AutoProcessor.from_pretrained(
            cfg.model.language_model.variant, 
            trust_remote_code=True, 
            use_fast=False
        )
    
    # Create data module (same as eval.py)
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=True,
        _recursive_=False
    )
    
    # Setup data module
    data_module.setup(stage='predict')
    
    # Get the predict dataset
    predict_dataset = data_module.predict_dataset
    
    print("\n" + "="*60)
    print("DREAMER DATASET STATISTICS")
    print("="*60)
    print(f"\nTotal number of data points: {len(predict_dataset)}")
    
    # Collect statistics by iterating through dataset
    mode_counter = Counter()
    allowed_counter = Counter()
    safe_to_execute_counter = Counter()
    mode_allowed_combinations = defaultdict(lambda: {"allowed": 0, "not_allowed": 0})
    mode_safety_combinations = defaultdict(lambda: {"safe": 0, "unsafe": 0})
    crash_mode_safety = {"safe_to_execute_true": 0, "safe_to_execute_false": 0, "safe_to_execute_none": 0}
    
    print("\nAnalyzing FULL dataset (this may take a while)...")
    print("Note: Each sample requires reading a gzipped JSON file, so this is disk I/O intensive.")
    
    # Analyze the full dataset
    sample_size = len(predict_dataset)
    
    for i in tqdm(range(sample_size), desc="Processing samples"):
        try:
            data_point = predict_dataset[i]
            
            # Get eval_infos which contains mode information
            eval_infos = data_point.eval_infos
            
            if eval_infos is not None:
                mode = eval_infos.get('mode', 'unknown')
                allowed = eval_infos.get('allowed', None)
                safe_to_execute = eval_infos.get('safe_to_execute', None)
                
                mode_counter[mode] += 1
                
                if allowed is not None:
                    allowed_counter[allowed] += 1
                    if allowed:
                        mode_allowed_combinations[mode]["allowed"] += 1
                    else:
                        mode_allowed_combinations[mode]["not_allowed"] += 1
                
                # Track safe_to_execute for all modes
                if safe_to_execute is not None:
                    safe_to_execute_counter[safe_to_execute] += 1
                    if safe_to_execute:
                        mode_safety_combinations[mode]["safe"] += 1
                    else:
                        mode_safety_combinations[mode]["unsafe"] += 1
                
                # Track safe_to_execute specifically for crash mode
                if mode == 'crash':
                    if safe_to_execute is True:
                        crash_mode_safety["safe_to_execute_true"] += 1
                    elif safe_to_execute is False:
                        crash_mode_safety["safe_to_execute_false"] += 1
                    else:
                        crash_mode_safety["safe_to_execute_none"] += 1
                
        except Exception as e:
            print(f"\nError processing data point {i}: {e}")
            continue
    
    # Print statistics
    print("\n" + "-"*60)
    print("MODE DISTRIBUTION")
    print("-"*60)
    
    total_modes = sum(mode_counter.values())
    for mode, count in sorted(mode_counter.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_modes) * 100 if total_modes > 0 else 0
        print(f"{mode:30s}: {count:6d} ({percentage:5.2f}%)")
    
    print("\n" + "-"*60)
    print("ALLOWED vs NOT ALLOWED")
    print("-"*60)
    
    total_allowed = sum(allowed_counter.values())
    for allowed_status, count in sorted(allowed_counter.items()):
        percentage = (count / total_allowed) * 100 if total_allowed > 0 else 0
        status_str = "ALLOWED" if allowed_status else "NOT ALLOWED"
        print(f"{status_str:30s}: {count:6d} ({percentage:5.2f}%)")
    
    print("\n" + "-"*60)
    print("MODE vs ALLOWED STATUS")
    print("-"*60)
    
    for mode in sorted(mode_allowed_combinations.keys()):
        allowed = mode_allowed_combinations[mode]["allowed"]
        not_allowed = mode_allowed_combinations[mode]["not_allowed"]
        total = allowed + not_allowed
        allowed_pct = (allowed / total) * 100 if total > 0 else 0
        not_allowed_pct = (not_allowed / total) * 100 if total > 0 else 0
        
        print(f"\n{mode}:")
        print(f"  Allowed:     {allowed:6d} ({allowed_pct:5.2f}%)")
        print(f"  Not Allowed: {not_allowed:6d} ({not_allowed_pct:5.2f}%)")
    
    print("\n" + "-"*60)
    print("CRASH MODE: safe_to_execute FIELD")
    print("-"*60)
    
    crash_total = sum(crash_mode_safety.values())
    if crash_total > 0:
        true_count = crash_mode_safety["safe_to_execute_true"]
        false_count = crash_mode_safety["safe_to_execute_false"]
        none_count = crash_mode_safety["safe_to_execute_none"]
        
        true_pct = (true_count / crash_total) * 100
        false_pct = (false_count / crash_total) * 100
        none_pct = (none_count / crash_total) * 100
        
        print(f"Total crash mode samples: {crash_total}")
        print(f"  safe_to_execute = True:  {true_count:6d} ({true_pct:5.2f}%)")
        print(f"  safe_to_execute = False: {false_count:6d} ({false_pct:5.2f}%)")
        print(f"  safe_to_execute = None:  {none_count:6d} ({none_pct:5.2f}%)")
    else:
        print("No crash mode samples found in dataset")
    
    print("\n" + "="*60)
    print(f"Analysis complete! (Analyzed all {sample_size} data points)")
    print("="*60)
    
    # Additional info
    print("\n" + "-"*60)
    print("DATASET CONFIGURATION")
    print("-"*60)
    print(f"Data path: {cfg.data_module.base_dataset.data_path}")
    if hasattr(cfg.data_module.base_dataset, 'filter_dreamer_mode'):
        print(f"Filter dreamer mode: {cfg.data_module.base_dataset.filter_dreamer_mode}")
    print(f"Use safety flag: {cfg.data_module.base_dataset.use_safety_flag}")
    print(f"Batch size: {cfg.data_module.batch_size}")
    print(f"Num workers: {cfg.data_module.num_workers}")
    

if __name__ == "__main__":
    main()

