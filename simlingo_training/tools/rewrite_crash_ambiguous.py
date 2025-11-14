#!/usr/bin/env python3
"""
Rewrite crash mode dreamer data with ambiguous instructions to mitigate causal confusion.

This script:
1. Loads existing crash samples from the dreamer dataset
2. Rewrites the instruction fields (dreamer_instruction, instructions_templates, templates_placeholders)
3. Keeps all other fields (trajectories, labels, etc.) unchanged
4. Saves to a new directory structure for seamless dataloader integration

Usage:
    python simlingo_training/tools/rewrite_crash_ambiguous.py --split train --limit 1000 --output_dir ambiguous_crash
"""

import argparse
import gzip
import json
import random
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple
import ujson
from tqdm import tqdm


# All 25 ambiguous templates
ALL_AMBIGUOUS_TEMPLATES = [
    # LOC + SPEED (1-10)
    "Drive to (x: <LOC_X> m, y: <LOC_Y> m) at <SPEED_MS> m/s.",
    "Proceed directly to (x: <LOC_X> m, y: <LOC_Y> m), maintaining <SPEED_MS> m/s.",
    "Head to the point (x: <LOC_X> m, y: <LOC_Y> m) with a speed of <SPEED_MS> m/s.",
    "Navigate to (x: <LOC_X> m, y: <LOC_Y> m) keeping <SPEED_MS> m/s.",
    "Reach the waypoint at (x: <LOC_X> m, y: <LOC_Y> m) at <SPEED_MS> m/s.",
    "Continue straight to (x: <LOC_X> m, y: <LOC_Y> m) at approximately <SPEED_MS> m/s.",
    "Advance toward (x: <LOC_X> m, y: <LOC_Y> m) without dropping below <SPEED_MS> m/s.",
    "Aim for (x: <LOC_X> m, y: <LOC_Y> m) while holding near <SPEED_MS> m/s.",
    "Drive to the coordinate (x: <LOC_X> m, y: <LOC_Y> m) using a constant speed of <SPEED_MS> m/s.",
    "Proceed to (x: <LOC_X> m, y: <LOC_Y> m), targeting <SPEED_MS> m/s near the point.",
    
    # OBJECT + SPEED (11-15)
    "Drive to the <OBJECT>'s position at <SPEED_MS> m/s.",
    "Proceed directly to the location of the <OBJECT>, maintaining <SPEED_MS> m/s.",
    "Reach the <OBJECT>'s coordinate at <SPEED_MS> m/s.",
    "Navigate to where the <OBJECT> is, keeping <SPEED_MS> m/s.",
    "Aim for the <OBJECT>'s point in space while holding near <SPEED_MS> m/s.",
    
    # LOC only (16-20)
    "Drive to (x: <LOC_X> m, y: <LOC_Y> m) keeping current speed.",
    "Continue to the coordinate (x: <LOC_X> m, y: <LOC_Y> m) at current pace.",
    "Proceed directly to (x: <LOC_X> m, y: <LOC_Y> m) without slowing down.",
    "Navigate to (x: <LOC_X> m, y: <LOC_Y> m) maintaining current velocity.",
    "Advance to the point at (x: <LOC_X> m, y: <LOC_Y> m).",
    
    # OBJECT only (21-25)
    "Drive to the <OBJECT>'s location keeping current speed.",
    "Proceed to where the <OBJECT> is without slowing.",
    "Continue directly to the <OBJECT>'s position.",
    "Navigate to the <OBJECT> at current pace.",
    "Reach the <OBJECT>'s coordinate without decelerating.",
]


def clean_object_name(object_type: str) -> str:
    """Clean object type name for natural language."""
    object_type = object_type.replace('_vqa', '').replace('crash_', '').replace('_', ' ')
    object_type = object_type.replace('static.prop.', '').replace('.', ' ')
    
    # Special case mappings
    mappings = {
        'constructioncone': 'construction cone',
        'warningconstruction': 'construction warning sign',
        'warningaccident': 'accident warning sign',
        'police': 'police car',
        'Sign_Yield': 'yield sign',
        'haybalelb': 'hay bale',
        'busstoplb': 'bus stop',
    }
    
    for key, value in mappings.items():
        if key in object_type:
            object_type = value
            break
    
    return object_type.strip()


def generate_ambiguous_instruction(
    crash_info: Dict,
    crash_templates: List[str],
    seed: int
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Generate ambiguous or explicit crash instruction.
    
    Returns:
        (dreamer_instruction, instructions_templates, templates_placeholders)
    """
    rng = random.Random(seed)
    
    # 80% ambiguous, 20% explicit
    use_ambiguous = rng.random() < 0.8
    
    if use_ambiguous:
        # Select random ambiguous template
        selected_template = rng.choice(ALL_AMBIGUOUS_TEMPLATES)
        
        # Extract values from crash_info
        crash_position = crash_info.get('crash_position', [0.0, 0.0, 0.0])
        loc_x = round(crash_position[0], 1)
        loc_y = round(crash_position[1], 1)
        
        # Get speed (try multiple fields)
        speed_ms = crash_info.get('target_speed')
        if speed_ms is None:
            speed_ms = crash_info.get('final_speed')
        if speed_ms is None:
            speed_ms = crash_info.get('crash_target_speed', 6.0)
        speed_ms = round(speed_ms, 1)
        
        # Get object name
        object_type = clean_object_name(crash_info.get('type', 'object'))
        
        # Fill placeholders in selected template
        filled_instruction = selected_template
        placeholders = {}
        
        if '<LOC_X>' in selected_template:
            filled_instruction = filled_instruction.replace('<LOC_X>', str(loc_x))
            placeholders['<LOC_X>'] = str(loc_x)
        
        if '<LOC_Y>' in selected_template:
            filled_instruction = filled_instruction.replace('<LOC_Y>', str(loc_y))
            placeholders['<LOC_Y>'] = str(loc_y)
        
        if '<SPEED_MS>' in selected_template:
            filled_instruction = filled_instruction.replace('<SPEED_MS>', str(speed_ms))
            placeholders['<SPEED_MS>'] = str(speed_ms)
        
        if '<OBJECT>' in selected_template:
            filled_instruction = filled_instruction.replace('<OBJECT>', object_type)
            placeholders['<OBJECT>'] = object_type
        
        # Return: random filled variant, ALWAYS first template, placeholders
        return (
            [filled_instruction],
            [ALL_AMBIGUOUS_TEMPLATES[0]],  # Always first template
            [placeholders]
        )
    
    else:
        # Use explicit crash template
        selected_template = rng.choice(crash_templates)
        object_type = clean_object_name(crash_info.get('type', 'object'))
        
        filled_instruction = selected_template.replace('<OBJECT>', object_type)
        
        return (
            [filled_instruction],
            [crash_templates[0]],  # Always first template
            [{'<OBJECT>': object_type}]
        )


def load_crash_templates() -> List[str]:
    """Load explicit crash templates from dreamer.json."""
    template_path = Path('data/augmented_templates/dreamer.json')
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    with open(template_path, 'r') as f:
        templates = ujson.load(f)
    
    return templates.get('crash', [])


def scan_dreamer_files(
    data_root: Path,
    split: str,
    skip_first_n_frames: int = 10,
    pred_len: int = 11,
    hist_len: int = 1,
    seed: int = 42,
    filter_crash_mode: bool = True
) -> List[Path]:
    """
    Scan and return dreamer files for crash mode rewriting.
    
    Generates for ALL routes in the specified split (train or eval).
    The dataloader will apply its own filtering (2% for eval, frame skipping, etc.) when loading.
    """
    # Get all route directories
    route_pattern = f"{data_root}/data/simlingo/*/*/*/Town*"
    route_dirs = glob.glob(route_pattern)
    print(f'Found {len(route_dirs)} total routes')
    
    # Filter by split only (no subsampling - generate for ALL routes)
    if split == "train":
        print("Using training routes (Town12)")
        route_dirs = [route_dir for route_dir in route_dirs if 'routes_training' in route_dir]
    elif split == "eval":
        print("Using validation routes (Town13)")
        route_dirs = [route_dir for route_dir in route_dirs if 'routes_validation' in route_dir]
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'eval'")
    
    print(f'Processing all {len(route_dirs)} routes (no route-level filtering applied)')
    
    # Collect all valid dreamer files with crash mode (if filtering enabled)
    dreamer_files = []
    
    for route_dir in tqdm(route_dirs, desc="Scanning routes for crash samples"):
        # Check if dreamer directory exists
        dreamer_dir = route_dir.replace('data/', 'dreamer/')
        if not os.path.exists(dreamer_dir):
            continue
        
        # Get RGB folder to determine number of frames
        rgb_folder = route_dir + '/rgb'
        if not os.path.exists(rgb_folder):
            continue
        
        num_seq = len(os.listdir(rgb_folder))
        
        # Iterate through valid frame range (same as dataset_base.py line 278)
        for seq in range(skip_first_n_frames, num_seq - pred_len - hist_len - 1):
            # Construct dreamer file path (same as dataset_base.py line 290)
            measurement_file = route_dir + '/measurements' + f'/{(seq + hist_len - 1):04}.json.gz'
            dreamer_file_path = measurement_file.replace('measurements', 'dreamer').replace('data/', 'dreamer/')
            
            if not os.path.exists(dreamer_file_path):
                continue
            
            # Pre-filter by crash mode (same as dataset_base.py lines 295-318)
            if filter_crash_mode:
                if has_crash_mode(Path(dreamer_file_path)):
                    dreamer_files.append(Path(dreamer_file_path))
            else:
                dreamer_files.append(Path(dreamer_file_path))
    
    print(f"Found {len(dreamer_files)} dreamer files with crash mode")
    return dreamer_files


def has_crash_mode(file_path: Path) -> bool:
    """Check if file contains any crash mode samples (pre-filter like dataset_base.py line 295)."""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = ujson.load(f)
        
        # Check if any options have crash mode (same logic as dataset_base.py lines 302-310)
        if isinstance(data, dict):
            for key, options in data.items():
                if 'factor' in key:
                    continue
                if isinstance(options, list):
                    for opt in options:
                        if opt.get('mode') == 'crash':
                            return True
        return False
    except Exception as e:
        return False


def extract_crash_samples(file_path: Path) -> List[Dict]:
    """Extract crash mode samples from a dreamer file."""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = ujson.load(f)
        
        crash_samples = []
        
        # Look for crash samples in the data
        if isinstance(data, dict):
            for mode_key, samples in data.items():
                if 'factor' in mode_key:
                    continue
                if isinstance(samples, list):
                    for sample in samples:
                        if isinstance(sample, dict) and sample.get('mode') == 'crash':
                            crash_samples.append(sample)
        
        return crash_samples
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def rewrite_crash_sample(
    crash_sample: Dict,
    crash_templates: List[str],
    seed: int
) -> Dict:
    """Rewrite a crash sample with ambiguous instruction."""
    # Generate new instruction fields
    dreamer_instruction, instructions_templates, templates_placeholders = \
        generate_ambiguous_instruction(
            crash_sample.get('info', {}),
            crash_templates,
            seed
        )
    
    # Create new sample with updated fields
    rewritten_sample = crash_sample.copy()
    rewritten_sample['dreamer_instruction'] = dreamer_instruction
    rewritten_sample['instructions_templates'] = instructions_templates
    rewritten_sample['templates_placeholders'] = templates_placeholders
    
    return rewritten_sample


def main():
    parser = argparse.ArgumentParser(
        description='Rewrite crash mode dreamer data with ambiguous instructions'
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'eval'],
        help='Dataset split to process'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of crash samples to rewrite (default: process all available)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ambiguous_crash',
        help='Output directory name (default: ambiguous_crash)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Paths
    data_root = Path('database/simlingo')
    output_root = Path(f'database/simlingo/{args.output_dir}')
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found at {data_root}")
    
    print(f"Loading crash templates...")
    crash_templates = load_crash_templates()
    print(f"Loaded {len(crash_templates)} explicit crash templates")
    
    print(f"\nScanning {args.split} split for crash samples...")
    print("Note: Generating for ALL routes. Dataloader will apply its own filtering when loading.")
    
    # Use same parameters as dataloader - includes crash mode filtering
    dreamer_files = scan_dreamer_files(
        data_root,
        args.split,
        skip_first_n_frames=10,
        pred_len=11,
        hist_len=1,
        seed=args.seed,
        filter_crash_mode=True  # Same as filter_dreamer_mode='crash' in dataloader
    )
    
    # Randomly shuffle files for sampling
    random.shuffle(dreamer_files)
    
    processed_count = 0
    skipped_count = 0
    
    # Determine total number to process
    if args.limit is not None:
        total_to_process = min(args.limit, len(dreamer_files))
        print(f"Processing up to {args.limit} crash samples...")
    else:
        total_to_process = len(dreamer_files)
        print(f"Processing all {total_to_process} available crash samples...")
    
    # Progress bar with known total
    pbar = tqdm(total=total_to_process, desc="Processing crash samples", unit="samples")
    
    for file_path in dreamer_files:
        if args.limit is not None and processed_count >= args.limit:
            break
        
        # Extract crash samples from this file
        crash_samples = extract_crash_samples(file_path)
        
        if not crash_samples:
            continue
        
        # Randomly select one crash sample from this file
        selected_crash = random.choice(crash_samples)
        
        # Check if we have required fields
        if 'info' not in selected_crash or 'crash_position' not in selected_crash.get('info', {}):
            skipped_count += 1
            pbar.set_postfix({'skipped': skipped_count})
            continue
        
        # Rewrite the crash sample
        rewritten_sample = rewrite_crash_sample(
            selected_crash,
            crash_templates,
            args.seed + processed_count  # Unique seed per sample
        )
        
        # Determine output path (maintain directory structure)
        # Convert from dreamer path to relative path
        file_str = str(file_path)
        if 'dreamer/simlingo' in file_str:
            relative_part = file_str.split('dreamer/simlingo', 1)[1].lstrip('/')
            relative_path = Path('simlingo') / relative_part
        else:
            # Fallback: try to extract relative path
            relative_path = file_path.relative_to(file_path.parents[7])
        
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as single crash sample per file
        output_data = {
            'crash': [rewritten_sample]
        }
        
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        
        processed_count += 1
        pbar.update(1)
        pbar.set_postfix({'skipped': skipped_count})
    
    pbar.close()
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Processed: {processed_count} crash samples")
    print(f"Skipped: {skipped_count} samples (missing required fields)")
    print(f"Output directory: {output_root}")
    print(f"{'='*60}")
    
    # Print sample statistics
    if processed_count > 0:
        ambiguous_count = int(processed_count * 0.8)
        explicit_count = processed_count - ambiguous_count
        print(f"\nExpected distribution (with seed={args.seed}):")
        print(f"  Ambiguous templates: ~{ambiguous_count} ({80}%)")
        print(f"  Explicit templates: ~{explicit_count} ({20}%)")
    else:
        print("\nNo crash samples were processed.")


if __name__ == '__main__':
    main()

