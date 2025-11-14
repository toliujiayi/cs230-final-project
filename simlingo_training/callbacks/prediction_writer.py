"""
Incremental prediction writer to avoid memory issues during evaluation.
Saves predictions to disk as they are generated, rather than accumulating in memory.
"""
import json
import datetime
from pathlib import Path
from typing import Any, List, Optional

import torch
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities import rank_zero_only


def decode_uint8(encoded: torch.Tensor) -> List[str]:
    return [row.tobytes().decode("utf-8").rstrip("\0") for row in encoded.cpu().numpy()]


def convert_to_json_serializable(obj):
    """Convert numpy arrays, torch tensors, and other objects to JSON-serializable format."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.cpu().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # For any other type, try to convert to string
        return str(obj)


class IncrementalPredictionWriter(BasePredictionWriter):
    """
    Writes predictions incrementally to disk to avoid accumulating everything in memory.
    Each batch's predictions are appended to JSONL files as they are generated.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.file_handles = {}
        self.batch_count = 0
        
    @rank_zero_only
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Setup output directory and files at the start of prediction."""
        # Determine output directory
        if self.output_dir is None:
            from hydra.utils import get_original_cwd
            repo_path = get_original_cwd()
            
            if trainer.ckpt_path is not None:
                ckpt_path = Path(trainer.ckpt_path).parent.parent
            else:
                model_variant = getattr(pl_module, 'language_model', None)
                if model_variant and hasattr(model_variant, 'variant'):
                    ckpt_path = Path(f'{repo_path}/outputs/{model_variant.variant}')
                else:
                    ckpt_path = Path(f'{repo_path}/outputs/default')
            
            self.output_dir = ckpt_path / "predictions"
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get rank for multi-GPU support
        self.rank = trainer.global_rank if trainer.world_size > 1 else 0
        
        # Create timestamped subdirectory for this prediction run
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.prediction_dir = self.output_dir / timestamp
        self.prediction_dir.mkdir(exist_ok=True, parents=True)
        
        # Open files for writing (JSONL format - one JSON object per line)
        self.files = {
            'waypoints': self.prediction_dir / f"waypoints_rank_{self.rank}.jsonl",
            'routes': self.prediction_dir / f"routes_rank_{self.rank}.jsonl",
            'language': self.prediction_dir / f"language_rank_{self.rank}.jsonl",
            'metadata': self.prediction_dir / f"metadata_rank_{self.rank}.jsonl",
        }
        
        # Open file handles
        for name, path in self.files.items():
            self.file_handles[name] = open(path, 'w')
        
        print(f"[Rank {self.rank}] Saving predictions to: {self.prediction_dir}")
        
    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[List[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Write predictions from a single batch to disk."""
        if prediction is None:
            return
        
        (speed_wps, route, language, speed_wps_gt, route_gt, language_gt, 
         run_ids, qa_templates, eval_infos, prompts) = prediction
        
        batch_size = len(run_ids)
        
        # Write each sample in the batch
        for i in range(batch_size):
            # Waypoints
            waypoint_data = {
                'run_id': run_ids[i],
                'batch_idx': batch_idx,
                'sample_idx': i,
                'prediction': convert_to_json_serializable(speed_wps[i]),
                'ground_truth': convert_to_json_serializable(speed_wps_gt[i]),
            }
            self.file_handles['waypoints'].write(json.dumps(waypoint_data) + '\n')
            
            # Routes
            route_data = {
                'run_id': run_ids[i],
                'batch_idx': batch_idx,
                'sample_idx': i,
                'prediction': convert_to_json_serializable(route[i]),
                'ground_truth': convert_to_json_serializable(route_gt[i]),
            }
            self.file_handles['routes'].write(json.dumps(route_data) + '\n')
            
            # Language
            language_data = {
                'run_id': run_ids[i],
                'batch_idx': batch_idx,
                'sample_idx': i,
                'prediction': language[i] if isinstance(language, list) else str(language[i]),
                'ground_truth': language_gt[i] if isinstance(language_gt, list) else str(language_gt[i]),
                'prompt': prompts[i] if isinstance(prompts, list) else str(prompts[i]),
            }
            self.file_handles['language'].write(json.dumps(language_data) + '\n')
            
            # Metadata (QA templates, eval_infos) - convert to JSON serializable
            qa_template_value = qa_templates[i] if isinstance(qa_templates, list) and i < len(qa_templates) else None
            eval_info_value = eval_infos[i] if isinstance(eval_infos, list) and i < len(eval_infos) else None
            
            metadata = {
                'run_id': run_ids[i],
                'batch_idx': batch_idx,
                'sample_idx': i,
                'qa_template': convert_to_json_serializable(qa_template_value),
                'eval_info': convert_to_json_serializable(eval_info_value),
            }
            self.file_handles['metadata'].write(json.dumps(metadata) + '\n')
        
        # Flush after each batch to ensure data is written
        for handle in self.file_handles.values():
            handle.flush()
        
        self.batch_count += 1
        if self.batch_count % 10 == 0:
            print(f"[Rank {self.rank}] Written {self.batch_count} batches ({self.batch_count * batch_size} samples)")
    
    @rank_zero_only
    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Close files and optionally consolidate results."""
        # Close all file handles
        for name, handle in self.file_handles.items():
            handle.close()
        
        print(f"\n[Rank {self.rank}] Prediction complete!")
        print(f"Total batches written: {self.batch_count}")
        print(f"Predictions saved to: {self.prediction_dir}")
        
        # Consolidate language predictions by category (COT, QA, ALL)
        self._consolidate_language_predictions()
        
        # Print post-processing instructions
        print("\n" + "="*60)
        print("To calculate metrics, run the post-processing script:")
        print(f"python simlingo_training/tools/postprocess_predictions.py \\")
        print(f"    --prediction_dir {self.prediction_dir}")
        print("="*60)
    
    @rank_zero_only
    def _consolidate_language_predictions(self):
        """Read language predictions and organize by category (cot, qa, all)."""
        language_file = self.files['language']
        metadata_file = self.files['metadata']
        
        if not language_file.exists() or not metadata_file.exists():
            return
        
        # Read all language predictions and metadata
        language_preds = []
        metadata = []
        
        with open(language_file, 'r') as f:
            for line in f:
                language_preds.append(json.loads(line))
        
        with open(metadata_file, 'r') as f:
            for line in f:
                metadata.append(json.loads(line))
        
        # Categorize by prompt type
        samples_cot = []
        samples_qa = []
        samples_all = []
        
        for i, (lang_pred, meta) in enumerate(zip(language_preds, metadata)):
            prompt = lang_pred['prompt']
            
            sample_data = (
                lang_pred['prediction'],
                lang_pred['ground_truth'],
                lang_pred['run_id']
            )
            
            samples_all.append(sample_data)
            
            if "What should the ego do next?" in prompt:
                samples_cot.append(sample_data)
            elif "Q:" in prompt:
                samples_qa.append(sample_data)
        
        # Save categorized predictions
        for samples, name in zip([samples_cot, samples_qa, samples_all], ["cot", "qa", "all"]):
            if len(samples) > 0:
                save_path = self.prediction_dir / f"language_preds_{name}_rank_{self.rank}.json"
                with open(save_path, 'w') as f:
                    json.dump(samples, f, indent=4)
                print(f"Saved {len(samples)} {name} predictions to {save_path}")
        
        # Handle QA template sorting if QA samples exist
        if len(samples_qa) > 0:
            sorted_samples = {}
            for i in [idx for idx, (l, _, _) in enumerate(samples_all) if (_, _, _) in samples_qa]:
                meta = metadata[i]
                if meta['qa_template']:
                    question, answer = meta['qa_template']
                    if question not in sorted_samples:
                        sorted_samples[question] = {}
                    if answer not in sorted_samples[question]:
                        sorted_samples[question][answer] = []
                    sorted_samples[question][answer].append(samples_all[i])
            
            if sorted_samples:
                save_path = self.prediction_dir / f"sorted_qa_templates_rank_{self.rank}.json"
                with open(save_path, 'w') as f:
                    json.dump(sorted_samples, f, indent=4)
                print(f"Saved sorted QA templates to {save_path}")

