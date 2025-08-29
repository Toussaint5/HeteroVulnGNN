import torch
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, Any, Optional
import torch.serialization
from torch.serialization import add_safe_globals


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.min_delta *= 1 if mode == 'min' else -1
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

class ModelCheckpoint:
    """Model checkpointing utility."""
    
    def __init__(self, save_dir: str = 'checkpoints', save_top_k: int = 1):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_top_k = save_top_k
        self.best_scores = []
        
    def save_checkpoint(self, state_dict: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{state_dict['epoch']}.pt"
        torch.save(state_dict, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(state_dict, best_path)
            
        # Keep only top k checkpoints
        if len(self.best_scores) >= self.save_top_k:
            # Remove worst checkpoint
            worst_score = max(self.best_scores)
            worst_idx = self.best_scores.index(worst_score)
            self.best_scores.pop(worst_idx)
            
            # Find and remove corresponding file
            for file_path in self.save_dir.glob("checkpoint_epoch_*.pt"):
                epoch = int(file_path.stem.split('_')[-1])
                if epoch not in [s[1] for s in self.best_scores]:
                    file_path.unlink()
                    break
        
        self.best_scores.append((state_dict['val_loss'], state_dict['epoch']))
        self.best_scores.sort()  # Sort by score
    
    def load_best_checkpoint(self) -> Dict:
        """Load best model checkpoint."""
        best_path = self.save_dir / "best_model.pt"
        if best_path.exists():
            # return torch.load(best_path, map_location='cpu')
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            add_safe_globals([np.dtype])
            return torch.load(best_path, map_location='cpu', weights_only=False)
        else:
            raise FileNotFoundError("No best model checkpoint found")
    
    def load_checkpoint(self, epoch: int) -> Dict:
        """Load specific epoch checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        if checkpoint_path.exists():
            return torch.load(checkpoint_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found")

class LearningRateScheduler:
    """Custom learning rate scheduler."""
    
    def __init__(self, optimizer, mode: str = 'reduce_on_plateau', **kwargs):
        self.optimizer = optimizer
        self.mode = mode
        
        if mode == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **kwargs
            )
        elif mode == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **kwargs
            )
        elif mode == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, **kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler mode: {mode}")
    
    def step(self, metric: Optional[float] = None):
        """Step the scheduler."""
        if self.mode == 'reduce_on_plateau':
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make torch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_predictions(predictions: Dict, targets: Dict, save_path: str):
    """Save model predictions and targets."""
    # Convert tensors to numpy for saving
    pred_numpy = {}
    target_numpy = {}
    
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            pred_numpy[key] = value.numpy()
        elif isinstance(value, dict):
            pred_numpy[key] = {k: v.numpy() for k, v in value.items()}
    
    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            target_numpy[key] = value.numpy()
        elif isinstance(value, dict):
            target_numpy[key] = {k: v.numpy() for k, v in value.items()}
    
    save_data = {
        'predictions': pred_numpy,
        'targets': target_numpy
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)

def load_predictions(load_path: str) -> tuple:
    """Load saved predictions and targets."""
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    return data['predictions'], data['targets']

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def memory_summary():
    """Print GPU memory summary if CUDA is available."""
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")