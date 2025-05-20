import os
import torch

def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, training_time, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'training_time': training_time,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict
    }
    print(f"Checkpoint saved to {save_path}, epoch {epoch}")
    torch.save(checkpoint, save_path)

def load_checkpoint(load_path, model, optimizer, scheduler):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")

    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    training_time = checkpoint['training_time']
    print(f"Checkpoint loaded from {load_path}, epoch {epoch}")
    
    return epoch, training_time
