import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch in pbar:
            optimizer.zero_grad()
            
            # Move batch to device
            src = batch['sequence'].to(device)  # [batch_size, seq_len, num_nucleotides]
            tgt = batch['structure'].to(device)  # [batch_size, seq_len]
            lengths = batch['length']
            
            # Create padding mask for variable length sequences
            padding_mask = create_padding_mask(lengths, src.size(1)).to(device)
            
            # Forward pass with teacher forcing during training
            # logits = model(
            #     src=src,
            #     tgt=tgt,
            #     src_key_padding_mask=padding_mask,
            #     tgt_key_padding_mask=padding_mask
            # )
            logits = model(src)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, num_classes]
            targets = tgt.view(-1)  # [batch_size * seq_len]
            
            # Create loss mask to ignore padded positions
            loss_mask = ~padding_mask.view(-1)
            
            # Calculate masked loss
            loss = criterion(logits[loss_mask], targets[loss_mask])
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update progress bar
            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss/(pbar.n+1):.4f}'
            })
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device, cal_metrics=False):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        with tqdm(val_loader, desc='Validation') as pbar:
            for batch in pbar:
                # Move batch to device
                src = batch['sequence'].to(device)
                tgt = batch['structure'].to(device)
                lengths = batch['length']
                
                # Create padding mask
                padding_mask = create_padding_mask(lengths, src.size(1)).to(device)
                
                # Forward pass (no teacher forcing during validation)
                # logits = model(
                #     src=src,
                #     src_key_padding_mask=padding_mask
                # )
                logits = model(src)
                # Get predictions
                predictions = logits.argmax(dim=-1)
                
                # Calculate loss
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = tgt.view(-1)
                loss_mask = ~padding_mask.view(-1)
                loss = criterion(logits_flat[loss_mask], targets_flat[loss_mask])
                
                # Store predictions and targets for metrics
                for pred, target, length in zip(predictions, tgt, lengths):
                    all_predictions.append(pred[:length].cpu())
                    all_targets.append(target[:length].cpu())
                
                # Update progress bar
                batch_loss = loss.item()
                total_loss += batch_loss
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{total_loss/(pbar.n+1):.4f}'
                })
    
    if cal_metrics:
        # Calculate additional metrics
        metrics = calculate_metrics(all_predictions, all_targets)
    else:
        metrics = {}
        
    return total_loss / num_batches, metrics

def create_padding_mask(lengths, max_len):
    """Create padding mask for variable length sequences"""
    batch_size = len(lengths)
    mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    return mask

def calculate_metrics(predictions, targets):
    """Calculate additional metrics for RNA structure prediction"""
    total_correct = 0
    total_positions = 0
    total_pairs_correct = 0
    total_pairs = 0
    
    for pred, target in zip(predictions, targets):
        # Accuracy for all positions
        total_correct += (pred == target).sum().item()
        total_positions += len(pred)
        
        # Base pair prediction accuracy
        pred_pairs = get_base_pairs(pred)
        target_pairs = get_base_pairs(target)
        total_pairs_correct += len(pred_pairs.intersection(target_pairs))
        total_pairs += len(target_pairs)
    
    metrics = {
        'accuracy': total_correct / total_positions,
        'pair_accuracy': total_pairs_correct / total_pairs if total_pairs > 0 else 0
    }
    
    return metrics

def get_base_pairs(structure):
    """Extract base pairs from structure labels"""
    pairs = set()
    stack = []
    
    for i, label in enumerate(structure):
        if label == "0":
            continue
        if label == 1:  # Opening bracket
            stack.append(i)
        elif label == 2 and stack:  # Closing bracket
            j = stack.pop()
            pairs.add((j, i))
    
    return pairs

# Example usage
def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Base Pair Accuracy: {metrics['pair_accuracy']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            
    return best_model