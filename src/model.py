import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LinearAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, chunk_size=512):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def chunk_compute_attention(self, q, k, v, chunk_size):
        batch_size, num_heads, seq_len, head_dim = q.shape
        chunks_q = q.split(chunk_size, dim=2)
        chunks_k = k.split(chunk_size, dim=2)
        chunks_v = v.split(chunk_size, dim=2)

        outputs = []
        for q_chunk in chunks_q:
            chunk_outputs = []
            for k_chunk, v_chunk in zip(chunks_k, chunks_v):
                # Compute attention scores for current chunks
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(
                    head_dim
                )
                attn = torch.softmax(scores, dim=-1)
                chunk_output = torch.matmul(attn, v_chunk)
                chunk_outputs.append(chunk_output)
            # Combine chunk outputs
            output = sum(chunk_outputs) / len(chunk_outputs)
            outputs.append(output)
        return torch.cat(outputs, dim=2)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # Generate Q, K, V with shape [batch, seq_len, 3 * hidden]
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention in chunks
        if seq_len > self.chunk_size:
            attn_output = self.chunk_compute_attention(q, k, v, self.chunk_size)
        else:
            scaling = float(self.head_dim) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scaling
            attn = torch.softmax(attn, dim=-1)
            attn_output = attn @ v

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, hidden_size
        )
        return self.dropout(self.proj(attn_output))


class EnhancedRnaEncoder(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        chunk_size=512,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        LinearAttention(hidden_size, num_heads, dropout, chunk_size),
                        nn.LayerNorm(hidden_size),
                        nn.Sequential(
                            nn.Linear(hidden_size, hidden_size * 2),  # Reduced from 4x
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_size * 2, hidden_size),
                            nn.Dropout(dropout),
                        ),
                        nn.LayerNorm(hidden_size),
                    ]
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = x.float()

        # Project input and add positional encoding
        x = checkpoint(self.input_proj, x)
        x = self.pos_encoding(x)
        x = self.layer_norm(x)

        # Process through attention layers with gradient checkpointing
        for attn, norm1, ffn, norm2 in self.layers:
            # Self-attention block with chunked computation
            attn_out = checkpoint(attn, x)
            x = norm1(x + attn_out)

            # Feed-forward block
            ffn_out = checkpoint(ffn, x)
            x = norm2(x + ffn_out)

        return x


class EnhancedRnaDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        output_size=61,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        chunk_size=512,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        current_size = hidden_size

        # Add attention layers except for the final output layer
        for i in range(num_layers - 1):
            self.layers.append(
                nn.ModuleList(
                    [
                        LinearAttention(current_size, num_heads, dropout, chunk_size),
                        nn.LayerNorm(current_size),
                        nn.Sequential(
                            nn.Linear(
                                current_size, current_size * 2
                            ),  # Reduced from 4x
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(current_size * 2, current_size),
                            nn.Dropout(dropout),
                        ),
                        nn.LayerNorm(current_size),
                    ]
                )
            )

        # Final output layer
        self.layers.append(nn.Sequential(nn.Linear(current_size, output_size)))

    def forward(self, x):
        # Process through attention layers with gradient checkpointing
        for i, layer in enumerate(self.layers[:-1]):
            attn, norm1, ffn, norm2 = layer

            # Self-attention block
            attn_out = checkpoint(attn, x)
            x = norm1(x + attn_out)

            # Feed-forward block
            ffn_out = checkpoint(ffn, x)
            x = norm2(x + ffn_out)

        # Final output layer
        x = self.layers[-1](x)
        return x


class EnhancedRNAPredictor(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        output_size=61,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        chunk_size=512,
        max_seq_length=5000,
    ):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.chunk_size = chunk_size

        self.encoder = EnhancedRnaEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            chunk_size=chunk_size,
        )

        self.decoder = EnhancedRnaDecoder(
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            chunk_size=chunk_size,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Check sequence length
        if x.size(1) > self.max_seq_length:
            raise ValueError(
                f"Input sequence length {x.size(1)} exceeds maximum allowed length {self.max_seq_length}"
            )

        if self.training:
            self.train()

        latent = self.encoder(x)
        logits = self.decoder(latent)
        return logits
