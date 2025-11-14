#!/usr/bin/env python3
"""
Simple script to convert language_rank_0.jsonl to language_preds_*_rank_0.json files.
Useful when prediction was interrupted before consolidation.

Usage:
    python consolidate_language.py language_rank_0.jsonl
    python consolidate_language.py /path/to/predictions/2025-11-12_19-09-09/language_rank_0.jsonl
"""

import argparse
import json
from pathlib import Path


def consolidate_language_predictions(jsonl_file: Path, rank: int = 0):
    """
    Convert JSONL language predictions to consolidated JSON files.
    
    JSONL format:
        {"run_id": "...", "prediction": "...", "ground_truth": "...", "prompt": "..."}
    
    Output format:
        [["prediction", "ground_truth", "run_id"], ...]
    """
    print(f"Loading language predictions from: {jsonl_file}")
    
    if not jsonl_file.exists():
        raise FileNotFoundError(f"File not found: {jsonl_file}")
    
    # Read JSONL file
    all_samples = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            sample = [
                data['prediction'],
                data['ground_truth'],
                data['run_id']
            ]
            all_samples.append({
                'data': sample,
                'prompt': data['prompt']
            })
    
    print(f"Loaded {len(all_samples)} samples")
    
    # Categorize by prompt type
    samples_cot = []
    samples_qa = []
    samples_all = []
    
    for item in all_samples:
        sample_data = item['data']
        prompt = item['prompt']
        
        samples_all.append(sample_data)
        
        if "What should the ego do next?" in prompt:
            samples_cot.append(sample_data)
        elif "Q:" in prompt:
            samples_qa.append(sample_data)
    
    # Determine output directory (same as input)
    output_dir = jsonl_file.parent
    
    # Save categorized predictions
    saved_files = []
    for samples, name in zip([samples_cot, samples_qa, samples_all], ["cot", "qa", "all"]):
        if len(samples) > 0:
            output_file = output_dir / f"language_preds_{name}_rank_{rank}.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=4)
            print(f"✓ Saved {len(samples)} {name} predictions to: {output_file.name}")
            saved_files.append(output_file)
        else:
            print(f"⊘ No {name} samples found, skipping...")
    
    print(f"\n✓ Consolidation complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Files created: {len(saved_files)}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert language JSONL to consolidated JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (from prediction directory)
  python consolidate_language.py language_rank_0.jsonl
  
  # Full path
  python consolidate_language.py outputs/simlingo/predictions/2025-11-12_19-09-09/language_rank_0.jsonl
  
  # Custom rank
  python consolidate_language.py language_rank_1.jsonl --rank 1

Output files created in same directory as input:
  - language_preds_cot_rank_0.json      (if COT samples exist)
  - language_preds_qa_rank_0.json       (if QA samples exist)  
  - language_preds_all_rank_0.json      (always created)
        """
    )
    
    parser.add_argument(
        'jsonl_file',
        type=str,
        help='Path to language_rank_N.jsonl file'
    )
    
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='GPU rank number (default: 0)'
    )
    
    args = parser.parse_args()
    
    jsonl_file = Path(args.jsonl_file)
    
    print("="*60)
    print("LANGUAGE PREDICTION CONSOLIDATION")
    print("="*60)
    print()
    
    consolidate_language_predictions(jsonl_file, rank=args.rank)
    
    print()
    print("="*60)


if __name__ == "__main__":
    main()

