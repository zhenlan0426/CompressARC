import os
import torch


def save_checkpoint(models, shared_optimizer, task_optimizers, epoch, checkpoint_path, continue_training=False):
    """
    Save model and optimizer states for continuing training
    
    Args:
        models: List of ARCCompressor models
        shared_optimizer: Optimizer for shared parameters
        task_optimizers: List of optimizers for task-specific parameters
        epoch: Current epoch number
        checkpoint_path: Path to save the checkpoint
        continue_training: If True, only save shared parameters and shared optimizer state
    """
    checkpoint = {
        'epoch': epoch,
        'shared_params': [p.detach().cpu() for p in models[0].shared_params],
    }
    
    if continue_training:
        checkpoint['shared_optimizer_state'] = shared_optimizer.state_dict()
        # Save task-specific parameters and optimizer states
        checkpoint['task_models'] = {}
        checkpoint['task_optimizers'] = {}
        
        for model, task_opt in zip(models, task_optimizers):
            task_name = model.multitensor_system.task.task_name
            checkpoint['task_models'][task_name] = [p.detach().cpu() for p in model.task_params]
            checkpoint['task_optimizers'][task_name] = task_opt.state_dict()    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path, models, shared_optimizer, task_optimizers=None, continue_training=False):
    """
    Load model and optimizer states to continue training
    
    Args:
        checkpoint_path: Path to the checkpoint file or loaded checkpoint dictionary
        models: List of ARCCompressor models
        shared_optimizer: Optimizer for shared parameters
        task_optimizers: List of optimizers for task-specific parameters (optional if continue_training checkpoint)
    
    Returns:
        start_epoch: The epoch to start training from
    """
    if isinstance(checkpoint_path, str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = checkpoint_path
        
    start_epoch = checkpoint['epoch'] + 1
    
    # Load shared parameters
    for p, saved in zip(models[0].shared_params if isinstance(models, list) else models.shared_params, \
                        checkpoint['shared_params']):
        p.data.copy_(saved.to(p.device, dtype=p.dtype))
    
    if continue_training:
        # Load shared optimizer state
        shared_optimizer.load_state_dict(checkpoint['shared_optimizer_state'])
        # Load task-specific parameters and optimizer states
        for model, task_opt in zip(models, task_optimizers):
            task_name = model.multitensor_system.task.task_name
            # Load task parameters
            for p, saved in zip(model.task_params, checkpoint['task_models'][task_name]):
                p.data.copy_(saved.to(p.device, dtype=p.dtype))
            # Load task optimizer state
            task_opt.load_state_dict(checkpoint['task_optimizers'][task_name])
    
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {start_epoch}")
    return start_epoch