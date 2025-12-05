import os
import gzip
import ujson
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from simlingo_training.config import TrainConfig


def analyze_training_data(dreamer_file_paths, desc="Processing training data"):
    """
    Analyze ALL options from dreamer training data files.
    Unlike the training dataset which randomly samples one option per file,
    this reads all options to get the true distribution.
    """
    mode_counter = Counter()
    allowed_counter = Counter()
    safe_to_execute_counter = Counter()
    mode_allowed_combinations = defaultdict(lambda: {"allowed": 0, "not_allowed": 0})
    mode_safety_combinations = defaultdict(lambda: {"safe": 0, "unsafe": 0})
    crash_mode_safety = {"safe_to_execute_true": 0, "safe_to_execute_false": 0, "safe_to_execute_none": 0}
    
    total_options = 0
    total_files = 0
    
    for file_path in tqdm(dreamer_file_paths, desc=desc):
        try:
            file_path_str = str(file_path, encoding='utf-8') if isinstance(file_path, bytes) else str(file_path)
            
            with gzip.open(file_path_str, 'rt') as f:
                alternative_trajectories = ujson.load(f)
            
            total_files += 1
            
            # Iterate through ALL options in this file (not just random sample)
            for key, options in alternative_trajectories.items():
                if 'factor' in key:
                    continue
                
                for option in options:
                    total_options += 1
                    
                    mode = option.get('mode', 'unknown')
                    allowed = option.get('allowed', None)
                    safe_to_execute = option.get('safe_to_execute', None)
                    
                    mode_counter[mode] += 1
                    
                    if allowed is not None:
                        allowed_counter[allowed] += 1
                        if allowed:
                            mode_allowed_combinations[mode]["allowed"] += 1
                        else:
                            mode_allowed_combinations[mode]["not_allowed"] += 1
                    
                    if safe_to_execute is not None:
                        safe_to_execute_counter[safe_to_execute] += 1
                        if safe_to_execute:
                            mode_safety_combinations[mode]["safe"] += 1
                        else:
                            mode_safety_combinations[mode]["unsafe"] += 1
                    
                    if mode == 'crash':
                        if safe_to_execute is True:
                            crash_mode_safety["safe_to_execute_true"] += 1
                        elif safe_to_execute is False:
                            crash_mode_safety["safe_to_execute_false"] += 1
                        else:
                            crash_mode_safety["safe_to_execute_none"] += 1
                            
        except Exception as e:
            print(f"\nError processing file {file_path}: {e}")
            continue
    
    return {
        'mode_counter': mode_counter,
        'allowed_counter': allowed_counter,
        'safe_to_execute_counter': safe_to_execute_counter,
        'mode_allowed_combinations': mode_allowed_combinations,
        'mode_safety_combinations': mode_safety_combinations,
        'crash_mode_safety': crash_mode_safety,
        'total_options': total_options,
        'total_files': total_files,
    }


def print_statistics(stats, dataset_type="TRAINING"):
    """Print formatted statistics."""
    mode_counter = stats['mode_counter']
    allowed_counter = stats['allowed_counter']
    safe_to_execute_counter = stats['safe_to_execute_counter']
    mode_allowed_combinations = stats['mode_allowed_combinations']
    mode_safety_combinations = stats['mode_safety_combinations']
    crash_mode_safety = stats['crash_mode_safety']
    
    print("\n" + "-"*60)
    print("MODE DISTRIBUTION")
    print("-"*60)
    
    total_modes = sum(mode_counter.values())
    for mode, count in sorted(mode_counter.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_modes) * 100 if total_modes > 0 else 0
        print(f"{mode:30s}: {count:8d} ({percentage:5.2f}%)")
    
    print("\n" + "-"*60)
    print("ALLOWED vs NOT ALLOWED")
    print("-"*60)
    
    total_allowed = sum(allowed_counter.values())
    for allowed_status, count in sorted(allowed_counter.items()):
        percentage = (count / total_allowed) * 100 if total_allowed > 0 else 0
        status_str = "ALLOWED" if allowed_status else "NOT ALLOWED"
        print(f"{status_str:30s}: {count:8d} ({percentage:5.2f}%)")
    
    print("\n" + "-"*60)
    print("SAFE TO EXECUTE DISTRIBUTION")
    print("-"*60)
    
    total_safe = sum(safe_to_execute_counter.values())
    for safe_status, count in sorted(safe_to_execute_counter.items()):
        percentage = (count / total_safe) * 100 if total_safe > 0 else 0
        status_str = "SAFE" if safe_status else "UNSAFE"
        print(f"{status_str:30s}: {count:8d} ({percentage:5.2f}%)")
    
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
        print(f"  Allowed:     {allowed:8d} ({allowed_pct:5.2f}%)")
        print(f"  Not Allowed: {not_allowed:8d} ({not_allowed_pct:5.2f}%)")
    
    print("\n" + "-"*60)
    print("MODE vs SAFE TO EXECUTE")
    print("-"*60)
    
    for mode in sorted(mode_safety_combinations.keys()):
        safe = mode_safety_combinations[mode]["safe"]
        unsafe = mode_safety_combinations[mode]["unsafe"]
        total = safe + unsafe
        safe_pct = (safe / total) * 100 if total > 0 else 0
        unsafe_pct = (unsafe / total) * 100 if total > 0 else 0
        
        print(f"\n{mode}:")
        print(f"  Safe:   {safe:8d} ({safe_pct:5.2f}%)")
        print(f"  Unsafe: {unsafe:8d} ({unsafe_pct:5.2f}%)")
    
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
        print(f"  safe_to_execute = True:  {true_count:8d} ({true_pct:5.2f}%)")
        print(f"  safe_to_execute = False: {false_count:8d} ({false_pct:5.2f}%)")
        print(f"  safe_to_execute = None:  {none_count:8d} ({none_pct:5.2f}%)")
    else:
        print("No crash mode samples found in dataset")


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42)
    
    # Parse command line arguments for analysis mode
    # Default to training data analysis
    ANALYZE_TRAINING = True  # Set to False to analyze validation data instead
    
    # Load checkpoint path config if specified
    load_path = 'outputs/simlingo/checkpoints/epoch=013.ckpt'
    if load_path is not None:
        load_path_config = Path(load_path).parent.parent / '.hydra/config.yaml'
        if load_path_config.exists():
            cfg = OmegaConf.load(load_path_config)
    
    # Set GPUs and workers
    cfg.gpus = 1
    cfg.data_module.num_workers = 8
    cfg.data_module.batch_size = 64
    
    print(f"Using {cfg.gpus} GPUs")
    
    if ANALYZE_TRAINING:
        print("\n" + "="*60)
        print("DREAMER TRAINING DATASET STATISTICS")
        print("="*60)
        print("\nAnalyzing TRAINING data (all options from dreamer files)")
        print("Note: This counts ALL options in each file, not just sampled ones")
        
        # Create the dreamer dataset to get file paths
        from simlingo_training.dataloader.dataset_dreamer import Data_Dreamer
        
        dreamer_dataset = Data_Dreamer(
            split="train",
            bucket_name="all",
            **cfg.data_module,
            **cfg.data_module.base_dataset,
        )
        
        print(f"\nTotal dreamer files (training): {len(dreamer_dataset)}")
        print(f"Dreamer folder: {cfg.data_module.base_dataset.get('dreamer_folder', 'dreamer')}")
        
        # Get the alternative_trajectories file paths
        dreamer_file_paths = dreamer_dataset.alternative_trajectories
        
        # Analyze all options from training files
        stats = analyze_training_data(dreamer_file_paths, desc="Processing training dreamer files")
        
        print("\n" + "="*60)
        print(f"TRAINING DATA SUMMARY")
        print("="*60)
        print(f"Total dreamer files: {stats['total_files']}")
        print(f"Total options (all modes combined): {stats['total_options']}")
        print(f"Average options per file: {stats['total_options'] / stats['total_files']:.1f}" if stats['total_files'] > 0 else "N/A")
        
        print_statistics(stats, "TRAINING")
        
        # Also analyze validation data for comparison
        print("\n\n" + "="*60)
        print("DREAMER VALIDATION DATASET STATISTICS")
        print("="*60)
        
        dreamer_val_dataset = Data_Dreamer(
            split="val",
            bucket_name="all",
            **cfg.data_module,
            **cfg.data_module.base_dataset,
        )
        
        print(f"\nTotal dreamer files (validation): {len(dreamer_val_dataset)}")
        
        val_file_paths = dreamer_val_dataset.alternative_trajectories
        val_stats = analyze_training_data(val_file_paths, desc="Processing validation dreamer files")
        
        print("\n" + "="*60)
        print(f"VALIDATION DATA SUMMARY")
        print("="*60)
        print(f"Total dreamer files: {val_stats['total_files']}")
        print(f"Total options (all modes combined): {val_stats['total_options']}")
        print(f"Average options per file: {val_stats['total_options'] / val_stats['total_files']:.1f}" if val_stats['total_files'] > 0 else "N/A")
        
        print_statistics(val_stats, "VALIDATION")
        
    else:
        # Original validation-only analysis using insteval_dataset
        from transformers import AutoProcessor, AutoTokenizer
        
        insteval_dataset = cfg.data_module.insteval_dataset
        
        print("\n" + "="*60)
        print("DREAMER EVAL DATASET STATISTICS (insteval_dataset)")
        print("="*60)
        
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.qa_dataset = None
        cfg.data_module.insteval_dataset = insteval_dataset
        cfg.data_module.base_dataset.use_safety_flag = True
        cfg.data_module.base_dataset.img_augmentation = False
        cfg.data_module.base_dataset.img_shift_augmentation = False
        
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
        
        data_module = hydra.utils.instantiate(
            cfg.data_module, 
            processor=processor,
            encoder_variant=cfg.model.vision_model.variant,
            llm_variant=cfg.model.language_model.variant,
            predict=True,
            _recursive_=False
        )
        
        data_module.setup(stage='predict')
        predict_dataset = data_module.predict_dataset
        
        print(f"\nTotal number of eval data points: {len(predict_dataset)}")
        
        # Original analysis code for eval dataset
        mode_counter = Counter()
        allowed_counter = Counter()
        safe_to_execute_counter = Counter()
        mode_allowed_combinations = defaultdict(lambda: {"allowed": 0, "not_allowed": 0})
        mode_safety_combinations = defaultdict(lambda: {"safe": 0, "unsafe": 0})
        crash_mode_safety = {"safe_to_execute_true": 0, "safe_to_execute_false": 0, "safe_to_execute_none": 0}
        
        for i in tqdm(range(len(predict_dataset)), desc="Processing eval samples"):
            try:
                data_point = predict_dataset[i]
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
                    
                    if safe_to_execute is not None:
                        safe_to_execute_counter[safe_to_execute] += 1
                        if safe_to_execute:
                            mode_safety_combinations[mode]["safe"] += 1
                        else:
                            mode_safety_combinations[mode]["unsafe"] += 1
                    
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
        
        stats = {
            'mode_counter': mode_counter,
            'allowed_counter': allowed_counter,
            'safe_to_execute_counter': safe_to_execute_counter,
            'mode_allowed_combinations': mode_allowed_combinations,
            'mode_safety_combinations': mode_safety_combinations,
            'crash_mode_safety': crash_mode_safety,
            'total_options': len(predict_dataset),
            'total_files': len(predict_dataset),
        }
        print_statistics(stats, "EVAL")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    
    # Configuration info
    print("\n" + "-"*60)
    print("DATASET CONFIGURATION")
    print("-"*60)
    print(f"Data path: {cfg.data_module.base_dataset.data_path}")
    print(f"Dreamer folder: {cfg.data_module.base_dataset.get('dreamer_folder', 'dreamer')}")
    if hasattr(cfg.data_module.base_dataset, 'filter_dreamer_mode'):
        print(f"Filter dreamer mode: {cfg.data_module.base_dataset.filter_dreamer_mode}")
    print(f"Use safety flag: {cfg.data_module.base_dataset.use_safety_flag}")
    

if __name__ == "__main__":
    main()

