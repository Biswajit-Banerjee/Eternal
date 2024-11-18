import torch
from .model import EnhancedRNAPredictor

def save_checkpoint(model, optimizer, epoch, path='checkpoints/checkpoint.pt'):
    """
    Simple function to save model checkpoint.
    
    Args:
        model: The RNA predictor model
        optimizer: The optimizer
        epoch: Current epoch number
        path: Where to save the checkpoint
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'model_config': {
            'input_size': model.encoder.input_proj.in_features,
            'hidden_size': model.encoder.input_proj.out_features,
            'output_size': model.decoder.layers[-1][0].out_features,
            'num_layers': len(model.encoder.layers),
            'num_heads': model.encoder.layers[0][0].num_heads,
            'dropout': model.encoder.layers[0][0].dropout.p,
            'chunk_size': model.chunk_size,
            'max_seq_length': model.max_seq_length
        }
    }, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path='checkpoints/checkpoint.pt', device='cuda'):
    """
    Simple function to load model checkpoint.
    
    Args:
        path: Path to the checkpoint
        device: Device to load the model to
        
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
        epoch: Epoch number when checkpoint was saved
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Create model from saved config
    config = checkpoint['model_config']
    model = EnhancedRNAPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        chunk_size=config['chunk_size'],
        max_seq_length=config['max_seq_length']
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create and load optimizer
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {path}")
    return model, optimizer, checkpoint['epoch']

# Example usage:
"""
# To save:
save_checkpoint(model, optimizer, epoch)

# To load:
model, optimizer, epoch = load_checkpoint()
"""