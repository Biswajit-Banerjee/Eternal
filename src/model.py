import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class RNATransformer(nn.Module):
    def __init__(
        self,
        num_nucleotides=5,
        num_structure_labels=3,
        d_model=128,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        # Embedding layers
        self.src_embed = nn.Linear(num_nucleotides, d_model)
        self.struct_embed = nn.Embedding(num_structure_labels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer
        self.output_layer = nn.Linear(d_model, num_structure_labels)

        # Loss function that ignores padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def create_mask(self, src_len, tgt_len, device):
        # Create source padding mask (not needed for this case as we use full sequences)
        src_mask = torch.zeros((src_len, src_len), device=device).bool()

        # Create target mask (traditional transformer subsequent mask)
        tgt_mask = torch.triu(
            torch.ones((tgt_len, tgt_len), device=device) * float("-inf"), diagonal=1
        )

        return src_mask, tgt_mask

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        device = src.device
        batch_size, seq_len, _ = src.shape

        # Embed input sequence
        src_embedded = self.src_embed(src)
        src_embedded = self.pos_encoder(src_embedded)

        if self.training and tgt is not None:
            # Teacher forcing
            tgt_embedded = self.struct_embed(tgt)
            tgt_embedded = self.pos_encoder(tgt_embedded)

            # Create masks
            src_mask, tgt_mask = self.create_mask(seq_len, seq_len, device)

            # Run through transformer
            output = self.transformer(
                src_embedded, tgt_embedded, src_mask=src_mask, tgt_mask=tgt_mask
            )

            # Project to structure labels
            output = self.output_layer(output)

        else:
            # Inference mode - generate structure auto-regressively
            output = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

            for i in range(seq_len):
                tgt = output[:, : i + 1]
                tgt_embedded = self.struct_embed(tgt)
                tgt_embedded = self.pos_encoder(tgt_embedded)

                # Create masks
                src_mask, tgt_mask = self.create_mask(seq_len, i + 1, device)

                # Generate next token
                out = self.transformer(
                    src_embedded, tgt_embedded, src_mask=src_mask, tgt_mask=tgt_mask
                )

                next_token = self.output_layer(out[:, -1:])
                next_token = next_token.argmax(dim=-1)

                output[:, i] = next_token.squeeze()

        return output

    def calculate_loss(self, predictions, targets):
        # Reshape predictions to (batch_size * seq_len, num_classes)
        predictions = predictions.view(-1, predictions.size(-1))
        targets = targets.view(-1)

        return self.criterion(predictions, targets)


class RNAStructurePredictor:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    def train_step(self, batch, optimizer):
        self.model.train()
        optimizer.zero_grad()

        # Move batch to device
        src = batch["sequence"].to(self.device)
        tgt = batch["structure"].to(self.device)

        # Forward pass
        output = self.model(src, tgt)

        # Calculate loss
        loss = self.model.calculate_loss(output, tgt)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        return loss.item()

    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            src = batch["sequence"].to(self.device)
            tgt = batch["structure"].to(self.device)

            output = self.model(src)
            loss = self.model.calculate_loss(output, tgt)

            predictions = output.argmax(dim=-1)

            return loss.item(), predictions

    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            src = sequence.to(self.device)
            output = self.model(src)
            return output.argmax(dim=-1)
