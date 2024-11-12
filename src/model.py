import torch
import torch.nn as nn

class RNAStructurePredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Bidirectional LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # * 2 for bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # 3 classes: '.', '(', ')'
        )

    def forward(self, x, lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process through LSTM
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply attention
        attn_output, _ = self.attention(
            output.transpose(0, 1),
            output.transpose(0, 1),
            output.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        
        # Generate predictions
        logits = self.fc(attn_output)
        return logits
    
