import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json

class MetricsTracker:
    def __init__(self, metrics_dir="metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.learning_rates = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        # self.per_class_metrics = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def update(self, train_loss, lr, test_metrics):
        self.train_losses.append(train_loss)
        self.learning_rates.append(lr)
        self.accuracies.append(test_metrics['accuracy'])
        self.precisions.append(test_metrics['precision'])
        self.recalls.append(test_metrics['recall'])
        self.f1_scores.append(test_metrics['f1'])
        # self.per_class_metrics.append(test_metrics['per_class'])
    
    def save_metrics(self):
        metrics_data = {
            "train_losses": self.train_losses,
            "learning_rates": self.learning_rates,
            "accuracies": self.accuracies,
            "precisions": self.precisions,
            "recalls": self.recalls,
            "f1_scores": self.f1_scores,
            # "per_class_metrics": self.per_class_metrics
        }
        
        # Save raw metrics
        metrics_file = self.metrics_dir / f"metrics_{self.timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f)
        
        # Create and save plots
        self._plot_training_metrics()
        self._plot_test_metrics()
    
    def _plot_training_metrics(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.metrics_dir / f"training_loss_{self.timestamp}.png")
        plt.close()
        
    def _plot_test_metrics(self):
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(self.accuracies) + 1)
        
        plt.plot(epochs, self.accuracies, label='Accuracy', marker='o')
        plt.plot(epochs, self.precisions, label='Precision', marker='s')
        plt.plot(epochs, self.recalls, label='Recall', marker='^')
        plt.plot(epochs, self.f1_scores, label='F1 Score', marker='D')
        
        plt.title('Test Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.metrics_dir / f"test_metrics_{self.timestamp}.png")
        plt.close()