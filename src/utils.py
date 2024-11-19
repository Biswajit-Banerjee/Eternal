import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

def inference(model, test_loader, device):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    # print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            sequences = batch['sequence'].to(device)
            structures = batch['structure'].to(device)
            lengths = batch['length'].to(device)
            
            # Forward pass
            logits = model(sequences)
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
            
            # Convert to flat arrays for evaluation, ignoring padding
            for pred, target, length in zip(preds, structures, lengths):
                # Only consider predictions up to actual sequence length
                pred = pred[:length].cpu().numpy()
                target = target[:length].cpu().numpy()
                
                # Ignore padded positions (target == -100)
                mask = target != -100
                all_preds.extend(pred[mask])
                all_targets.extend(target[mask])
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, 
        all_preds, 
        average='weighted'
    )
    
    # Calculate per-class metrics
    per_class_metrics = {}
    labels = ['.', '(', ')']
    class_precision, class_recall, class_f1, support = precision_recall_fscore_support(
        all_targets, 
        all_preds
    )
    
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': class_precision[i],
            'recall': class_recall[i],
            'f1': class_f1[i],
            'support': support[i]
        }
    
    # print("\nOverall Metrics:")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    
    # print("\nPer-Class Metrics:")
    # for label, metrics in per_class_metrics.items():
    #     print(f"\nClass {label}:")
    #     print(f"Precision: {metrics['precision']:.4f}")
    #     print(f"Recall: {metrics['recall']:.4f}")
    #     print(f"F1 Score: {metrics['f1']:.4f}")
    #     print(f"Support: {metrics['support']}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        # 'per_class': per_class_metrics
    }

def visualize_predictions(model, test_loader, device, num_examples=1, total_examples=5):
    model.eval()
    total = 0
    print(f"\nVisualizing {num_examples} example predictions:")
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            structures = batch['structure'].to(device)
            lengths = batch['length'].to(device)
            true_structures = batch['raw_structure']
            true_sequence = batch['raw_sequence']
            
            # Forward pass
            logits = model(sequences)
            preds = torch.argmax(logits, dim=-1)
            
            # Convert indices back to structure notation
            idx_to_struct = {v:k for k,v in test_loader.dataset.struct_to_idx.items()}
            
            for i in range(min(num_examples, len(sequences))):
                length = lengths[i].item()
                
                # Convert prediction to structure string
                pred_struct = ''.join([
                    idx_to_struct[idx.item()] 
                    for idx in preds[i][:length]
                ])
                
                # Convert ground truth to structure string
                true_struct = true_structures[i]
                
                # Convert one-hot sequence back to string
                seq = true_sequence[i]
                
                print(f"\nExample {total+1:3}:")
                print(f"Sequence:  {seq}")
                print(f"Predicted: {pred_struct}")
                print(f"Ground Tr: {true_struct}")
                total += 1
                
                if total >= total_examples:
                    return
            
