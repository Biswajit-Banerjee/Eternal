import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch in pbar:
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch['sequences'], batch['lengths'])
            
            # Reshape for loss calculation
            logits = logits.view(-1, 3)  # (batch_size * seq_len, num_classes)
            targets = batch['structures'].view(-1)  # (batch_size * seq_len)
            
            # Calculate loss
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        with tqdm(val_loader, desc='Validation') as pbar:
            for batch in pbar:
                # Forward pass
                logits = model(batch['sequences'], batch['lengths'])
                
                # Reshape for loss calculation
                logits = logits.view(-1, 3)
                targets = batch['structures'].view(-1)
                
                # Calculate loss
                loss = criterion(logits, targets)
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches