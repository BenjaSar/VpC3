"""
Multi-stage Learning Rate Schedulers
Implements warmup, cosine annealing, and fine-tuning stages
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class MultiStageLRScheduler(_LRScheduler):
    """
    Multi-stage learning rate scheduler with:
    1. Warmup stage: Linear increase from min_lr to max_lr
    2. Cosine annealing stage: Gradual decrease with cosine function
    3. Fine-tuning stage: Minimal learning rate for refinement
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        cosine_epochs: Number of epochs for cosine annealing
        finetune_epochs: Number of epochs for fine-tuning
        max_lr: Maximum learning rate (reached after warmup)
        min_lr: Minimum learning rate (for fine-tuning)
        last_epoch: The index of last epoch
    """
    def __init__(self, optimizer, warmup_epochs=5, cosine_epochs=45, 
                 finetune_epochs=50, max_lr=1e-3, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = cosine_epochs
        self.finetune_epochs = finetune_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = warmup_epochs + cosine_epochs + finetune_epochs
        
        super(MultiStageLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate based on current epoch"""
        epoch = self.last_epoch
        
        if epoch < self.warmup_epochs:
            # Warmup stage: Linear increase
            lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
            
        elif epoch < self.warmup_epochs + self.cosine_epochs:
            # Cosine annealing stage
            cosine_epoch = epoch - self.warmup_epochs
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * cosine_epoch / self.cosine_epochs)
            )
            
        else:
            # Fine-tuning stage: Minimal LR
            lr = self.min_lr
        
        return [lr for _ in self.base_lrs]


class WarmupCosineLR(_LRScheduler):
    """
    Warmup + Cosine Annealing scheduler (simplified version)
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        min_lr: Minimum learning rate
        last_epoch: The index of last epoch
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate"""
        epoch = self.last_epoch
        
        if epoch < self.warmup_epochs:
            # Warmup stage
            alpha = epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_epoch = epoch - self.warmup_epochs
            cosine_total = self.total_epochs - self.warmup_epochs
            
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                    1 + np.cos(np.pi * cosine_epoch / cosine_total)
                )
                for base_lr in self.base_lrs
            ]


class PolyLR(_LRScheduler):
    """
    Polynomial learning rate decay
    Common in semantic segmentation tasks
    
    Args:
        optimizer: PyTorch optimizer
        total_epochs: Total number of training epochs
        power: Polynomial power (default: 0.9)
        min_lr: Minimum learning rate
        last_epoch: The index of last epoch
    """
    def __init__(self, optimizer, total_epochs, power=0.9, min_lr=1e-6, last_epoch=-1):
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate"""
        factor = (1 - self.last_epoch / self.total_epochs) ** self.power
        return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]


def get_scheduler(optimizer, config):
    """
    Factory function to create scheduler based on config
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary with scheduler parameters
    
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.get('type', 'multistage')
    
    if scheduler_type == 'multistage':
        return MultiStageLRScheduler(
            optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            cosine_epochs=config.get('cosine_epochs', 45),
            finetune_epochs=config.get('finetune_epochs', 50),
            max_lr=config.get('max_lr', 1e-3),
            min_lr=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'warmup_cosine':
        return WarmupCosineLR(
            optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            total_epochs=config.get('total_epochs', 100),
            min_lr=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'poly':
        return PolyLR(
            optimizer,
            total_epochs=config.get('total_epochs', 100),
            power=config.get('power', 0.9),
            min_lr=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('total_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class GradientMonitor:
    """
    Monitor gradient statistics during training
    Helps detect vanishing/exploding gradients
    """
    def __init__(self, model, log_interval=100):
        """
        Args:
            model: PyTorch model
            log_interval: Log gradient stats every N steps
        """
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        
    def check_gradients(self):
        """
        Check gradient statistics
        
        Returns:
            Dictionary with gradient statistics
        """
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        num_params = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                min_grad = min(min_grad, p.grad.data.abs().min().item())
                num_params += 1
        
        total_norm = total_norm ** 0.5
        
        return {
            'total_norm': total_norm,
            'max_grad': max_grad,
            'min_grad': min_grad,
            'num_params_with_grad': num_params
        }
    
    def log_gradients(self, writer=None, global_step=None):
        """
        Log gradient statistics to tensorboard
        
        Args:
            writer: TensorBoard SummaryWriter
            global_step: Global training step
        """
        self.step_count += 1
        
        if self.step_count % self.log_interval != 0:
            return
        
        grad_stats = self.check_gradients()
        
        if writer is not None and global_step is not None:
            writer.add_scalar('gradients/total_norm', grad_stats['total_norm'], global_step)
            writer.add_scalar('gradients/max_grad', grad_stats['max_grad'], global_step)
            writer.add_scalar('gradients/min_grad', grad_stats['min_grad'], global_step)
        
        # Check for potential issues
        if grad_stats['total_norm'] > 100:
            print(f"Warning: Large gradient norm detected: {grad_stats['total_norm']:.2f}")
        
        if grad_stats['max_grad'] < 1e-7:
            print(f"Warning: Very small gradients detected: {grad_stats['max_grad']:.2e}")
        
        return grad_stats
