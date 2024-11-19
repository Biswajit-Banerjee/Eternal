import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import gc
import logging
from tqdm import tqdm
import math
import json
import matplotlib.pyplot as plt
from datetime import datetime

# local imports
from src.data import RNADataset
from src.model import EnhancedRNAPredictor
from src.train import train_epoch, validate_epoch, train_model, create_padding_mask
from src.utils import inference, visualize_predictions
from src.objective import create_criterion
from src.checkpoint import save_checkpoint, load_checkpoint
from src.metrics import MetricsTracker
import warnings

warnings.filterwarnings("ignore")

# Set up logging
def setup_logging(log_dir="logs"):
    """Set up logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return timestamp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
batch_size = 28
num_epochs = 3
learning_rate = 0.003
model_save_dir = Path("models")

# Set up logging and metrics tracking
timestamp = setup_logging()
# metrics_tracker = MetricsTracker()

# Log basic configuration
logging.info(f"Starting training with configuration:")
logging.info(f"Batch size: {batch_size}")
logging.info(f"Number of epochs: {num_epochs}")
logging.info(f"Learning rate: {learning_rate}")
logging.info(f"Device: {device}")

# Create model save directory
model_save_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
logging.info("Loading datasets...")
train_dataset = RNADataset.load("data/val_dataset.pkl")
test_dataset = RNADataset.load("data/test_dataset.pkl")

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

logging.info(f"Train data batches: {len(train_dataset)}")
# logging.info(f"Validation data batches: {len(val_loader)}")
logging.info(f"Test data batches: {len(test_dataset)}")

# Initialize model
logging.info("Initializing model...")
model = EnhancedRNAPredictor(
    input_size=5, hidden_size=32, output_size=61, num_layers=3, num_heads=8, dropout=0.1
).to(device)

# Set up training components
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)

best_loss = float("inf")
logging.info(f"Criterion: {criterion}")
logging.info("Starting training loop...")
for epoch in range(num_epochs):
    # Clear cache before each epoch
    gc.collect()
    torch.cuda.empty_cache()

    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device=device, test_loader=test_loader)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Run inference on test set
    test_metrics = inference(model, test_loader, device)
    
    # Log metrics
    logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
    logging.info(f"Training Loss: {train_loss:.4f}")
    logging.info(f"Learning Rate: {current_lr}")
    logging.info(f"Test Metrics:")
    logging.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logging.info(f"  Precision: {test_metrics['precision']:.4f}")
    logging.info(f"  Recall: {test_metrics['recall']:.4f}")
    logging.info(f"  F1 Score: {test_metrics['f1']:.4f}")

    # Store metrics
    # metrics_tracker.update(train_loss, current_lr, test_metrics)

    # Update learning rate
    scheduler.step(train_loss)
    save_checkpoint(model, optimizer, epoch)

    # Save final metrics and create visualizations
    # metrics_tracker.save_metrics()

    # save model
    if best_loss > train_loss:
        best_loss = train_loss
        save_checkpoint(model, optimizer, epoch, "checkpoints/best.pt")
        logging.info(f"New best model saved with loss: {best_loss:.4f}")


logging.info("Training completed.")