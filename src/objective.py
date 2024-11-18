import torch
import torch.nn as nn
import torch.nn.functional as F


class BracketBalancedCrossEntropyLoss(nn.Module):
    def __init__(self, struct_to_idx, ignore_index=-100, bracket_weight=0.5):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )
        self.bracket_weight = bracket_weight

        # Store structure to index mapping
        self.struct_to_idx = struct_to_idx

        # Define bracket pairs
        self.openings = list("({[<") + [chr(ord("A") + i) for i in range(26)]
        self.closings = list(">]})") + [chr(ord("a") + i) for i in range(26)]

        # Get indices for all brackets
        self.opening_indices = [
            self.struct_to_idx[b] for b in self.openings if b in self.struct_to_idx
        ]
        self.closing_indices = [
            self.struct_to_idx[b] for b in self.closings if b in self.struct_to_idx
        ]

    def compute_bracket_balance_loss(self, logits, targets):
        """
        Compute loss based on balance of all bracket types
        """
        # Get probabilities for all positions
        probs = F.softmax(logits, dim=-1)  # shape: (batch, seq_len, num_classes)

        # Create mask for valid positions
        mask = (targets != -100).float()  # shape: (batch, seq_len)
        mask = mask.unsqueeze(-1)  # shape: (batch, seq_len, 1)

        # Sum probabilities for all opening brackets
        total_open = torch.zeros_like(mask[:, 0, 0])  # shape: (batch,)
        for idx in self.opening_indices:
            total_open += (probs[..., idx] * mask.squeeze(-1)).sum(dim=1)

        # Sum probabilities for all closing brackets
        total_close = torch.zeros_like(mask[:, 0, 0])  # shape: (batch,)
        for idx in self.closing_indices:
            total_close += (probs[..., idx] * mask.squeeze(-1)).sum(dim=1)

        # Compute balance loss (should be close to 0)
        balance_loss = torch.abs(total_open - total_close)

        # Normalize by number of valid positions
        valid_positions = mask.squeeze(-1).sum(dim=1) + 1e-8
        balance_loss = balance_loss / valid_positions

        return balance_loss.mean()

    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits (B, C) where B is batch size * sequence length
                    and C is number of classes
            targets: Target indices (B,) where B is batch size * sequence length
        """
        # Compute standard cross entropy loss
        ce_loss = self.cross_entropy(logits, targets)

        # Reshape tensors back to (batch_size, seq_len) for bracket loss
        batch_size = 1  # Since we're getting flattened input
        seq_len = ce_loss.size(0)
        logits_reshaped = logits.view(batch_size, seq_len, -1)
        targets_reshaped = targets.view(batch_size, seq_len)

        # Compute mean CE loss for non-ignored positions
        mask = (targets != -100).float()
        ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)

        # Compute bracket balance loss
        bracket_loss = self.compute_bracket_balance_loss(
            logits_reshaped, targets_reshaped
        )

        # Combine losses
        total_loss = ce_loss + self.bracket_weight * bracket_loss

        return total_loss


def create_criterion(struct_to_idx, ignore_index=-100, bracket_weight=0.5):
    """Create the bracket-aware criterion"""
    return BracketBalancedCrossEntropyLoss(
        struct_to_idx=struct_to_idx,
        ignore_index=ignore_index,
        bracket_weight=bracket_weight,
    )
