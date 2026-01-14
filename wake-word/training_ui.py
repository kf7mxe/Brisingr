#!/usr/bin/env python3
"""
Wake Word Training UI - Graphical interface for training wake word detection models
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import sys
import os
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Import model classes from training-v3.py
sys.path.insert(0, str(Path(__file__).parent / 'training'))

try:
    from training.training_v3 import (
        TinyWakeWordCNN,
        UltraTinyWakeWordCNN,
        MobileNetStyleWakeWord,
        QuantizedFriendlyWakeWord,
        OptimizedWakeWordDataset
    )
except ImportError:
    # Define minimal versions if import fails
    class TinyWakeWordCNN(nn.Module):
        def __init__(self, input_dim=13, sequence_length=101):
            super().__init__()
            self.input_dim = input_dim
            self.sequence_length = sequence_length
            self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(32)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(32, 2)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.global_pool(x).squeeze(-1)
            x = self.dropout(x)
            x = self.fc(x)
            return F.log_softmax(x, dim=1)

    class UltraTinyWakeWordCNN(nn.Module):
        def __init__(self, input_dim=13, sequence_length=101):
            super().__init__()
            self.input_dim = input_dim
            self.sequence_length = sequence_length
            self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1)
            self.gn1 = nn.GroupNorm(4, 16)
            self.gn2 = nn.GroupNorm(2, 8)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = F.relu6(self.gn1(self.conv1(x)))
            x = F.relu6(self.gn2(self.conv2(x)))
            x = self.global_pool(x).squeeze(-1)
            x = self.fc(x)
            return F.log_softmax(x, dim=1)

    class OptimizedWakeWordDataset(Dataset):
        def __init__(self, x, y, target_length=None, augment=False):
            self.x = torch.FloatTensor(x)
            self.y = torch.LongTensor(y)
            self.augment = augment
            self.target_length = target_length

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            x = self.x[idx]
            y = self.y[idx]

            if self.target_length and x.shape[0] != self.target_length:
                if x.shape[0] > self.target_length:
                    if self.augment:
                        start = torch.randint(0, x.shape[0] - self.target_length + 1, (1,))
                        x = x[start:start + self.target_length]
                    else:
                        start = (x.shape[0] - self.target_length) // 2
                        x = x[start:start + self.target_length]
                else:
                    pad_size = self.target_length - x.shape[0]
                    x = F.pad(x, (0, 0, 0, pad_size))

            if self.augment and torch.rand(1) < 0.2:
                noise = torch.randn_like(x) * 0.005
                x = x + noise

            return x, y


class TrainingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Wake Word Training UI")
        self.root.geometry("900x700")

        # Training state
        self.training_active = False
        self.log_queue = queue.Queue()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Model architectures
        self.model_classes = {
            "TinyWakeWordCNN (Standard)": TinyWakeWordCNN,
            "UltraTinyWakeWordCNN (Efficient)": UltraTinyWakeWordCNN,
            "MobileNetStyleWakeWord": None,  # Available if imported
            "QuantizedFriendlyWakeWord": None
        }

        # Create UI
        self.create_ui()

        # Start log processor
        self.process_logs()

    def create_ui(self):
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Data & Model Configuration
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="Configuration")
        self.create_config_tab(config_frame)

        # Tab 2: Training Progress
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="Training")
        self.create_training_tab(training_frame)

        # Tab 3: Results
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        self.create_results_tab(results_frame)

    def create_config_tab(self, parent):
        # Dataset Selection
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Configuration", padding=10)
        dataset_frame.pack(fill=tk.X, padx=10, pady=5)

        # Data file selection
        ttk.Label(dataset_frame, text="Training Data File (.npz):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.data_file_var = tk.StringVar(value="optimized_wake_word_data.npz")
        ttk.Entry(dataset_frame, textvariable=self.data_file_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_data_file).grid(row=0, column=2)

        # Or select individual directories
        ttk.Label(dataset_frame, text="OR select individual directories:").grid(row=1, column=0, columnspan=3, pady=(10, 5))

        # Positive samples directory
        ttk.Label(dataset_frame, text="Positive Samples Dir:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.positive_dir_var = tk.StringVar()
        ttk.Entry(dataset_frame, textvariable=self.positive_dir_var, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_positive_dir).grid(row=2, column=2)

        # Negative samples directory
        ttk.Label(dataset_frame, text="Negative Samples Dir:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.negative_dir_var = tk.StringVar()
        ttk.Entry(dataset_frame, textvariable=self.negative_dir_var, width=50).grid(row=3, column=1, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_negative_dir).grid(row=3, column=2)

        # Model Configuration
        model_frame = ttk.LabelFrame(parent, text="Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        # Model architecture selection
        ttk.Label(model_frame, text="Model Architecture:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="UltraTinyWakeWordCNN (Efficient)")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                    values=list(self.model_classes.keys()),
                                    state="readonly", width=40)
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Training Parameters
        params_frame = ttk.LabelFrame(parent, text="Training Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(params_frame, from_=1, to=500, textvariable=self.epochs_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.batch_size_var = tk.IntVar(value=64)
        ttk.Spinbox(params_frame, from_=8, to=256, textvariable=self.batch_size_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.DoubleVar(value=0.002)
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)

        # Output model path
        ttk.Label(params_frame, text="Output Model Path:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_path_var = tk.StringVar(value="ultra_tiny_wake_word.pth")
        ttk.Entry(params_frame, textvariable=self.output_path_var, width=40).grid(row=3, column=1, padx=5)
        ttk.Button(params_frame, text="Browse", command=self.browse_output_path).grid(row=3, column=2)

        # Device info
        device_text = f"Using device: {self.device}"
        ttk.Label(params_frame, text=device_text, foreground="blue").grid(row=4, column=0, columnspan=3, pady=10)

    def create_training_tab(self, parent):
        # Progress frame
        progress_frame = ttk.Frame(parent, padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        ttk.Label(progress_frame, text="Training Progress:").pack(anchor=tk.W, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Current epoch/status
        self.status_var = tk.StringVar(value="Ready to train")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, font=("Arial", 10, "bold"))
        status_label.pack(anchor=tk.W, pady=5)

        # Log output
        ttk.Label(progress_frame, text="Training Log:").pack(anchor=tk.W, pady=5)
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=20, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Control buttons
        button_frame = ttk.Frame(progress_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.train_button = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Clear Log", command=lambda: self.log_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)

    def create_results_tab(self, parent):
        results_frame = ttk.Frame(parent, padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(results_frame, text="Training Results", font=("Arial", 14, "bold")).pack(pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=25, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.results_text.insert(tk.END, "Training results will appear here after training completes.\n\n")
        self.results_text.insert(tk.END, "Metrics included:\n")
        self.results_text.insert(tk.END, "- Training loss per epoch\n")
        self.results_text.insert(tk.END, "- Validation accuracy\n")
        self.results_text.insert(tk.END, "- Test accuracy\n")
        self.results_text.insert(tk.END, "- Model parameters\n")
        self.results_text.config(state=tk.DISABLED)

    def browse_data_file(self):
        filename = filedialog.askopenfilename(
            title="Select Training Data File",
            filetypes=[("NumPy Archive", "*.npz"), ("All Files", "*.*")]
        )
        if filename:
            self.data_file_var.set(filename)

    def browse_positive_dir(self):
        dirname = filedialog.askdirectory(title="Select Positive Samples Directory")
        if dirname:
            self.positive_dir_var.set(dirname)

    def browse_negative_dir(self):
        dirname = filedialog.askdirectory(title="Select Negative Samples Directory")
        if dirname:
            self.negative_dir_var.set(dirname)

    def browse_output_path(self):
        filename = filedialog.asksaveasfilename(
            title="Save Model As",
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            self.output_path_var.set(filename)

    def log(self, message):
        """Thread-safe logging"""
        self.log_queue.put(message)

    def process_logs(self):
        """Process log messages from queue"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_logs)

    def start_training(self):
        # Validate inputs
        data_file = self.data_file_var.get()
        if not os.path.exists(data_file):
            messagebox.showerror("Error", f"Data file not found: {data_file}")
            return

        # Start training in background thread
        self.training_active = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        thread = threading.Thread(target=self.train_model, daemon=True)
        thread.start()

    def stop_training(self):
        self.training_active = False
        self.status_var.set("Stopping training...")
        self.log("Training stopped by user")

    def train_model(self):
        """Main training logic (runs in background thread)"""
        try:
            self.log("="*60)
            self.log("Starting Wake Word Training")
            self.log("="*60)

            # Load data
            self.status_var.set("Loading data...")
            self.log(f"Loading data from: {self.data_file_var.get()}")

            data = np.load(self.data_file_var.get())
            x_train = data['x_train']
            y_train = data['y_train']
            x_val = data['x_val']
            y_val = data['y_val']
            x_test = data['x_test']
            y_test = data['y_test']

            self.log(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

            # Create model
            model_name = self.model_var.get()
            model_class = self.model_classes[model_name]
            if model_class is None:
                raise ValueError(f"Model {model_name} not available")

            input_dim = x_train.shape[2]
            sequence_length = x_train.shape[1]

            self.log(f"Creating model: {model_name}")
            model = model_class(input_dim=input_dim, sequence_length=sequence_length).to(self.device)

            total_params = sum(p.numel() for p in model.parameters())
            self.log(f"Model parameters: {total_params:,}")

            # Create datasets
            train_dataset = OptimizedWakeWordDataset(x_train, y_train, target_length=sequence_length, augment=True)
            val_dataset = OptimizedWakeWordDataset(x_val, y_val, target_length=sequence_length, augment=False)
            test_dataset = OptimizedWakeWordDataset(x_test, y_test, target_length=sequence_length, augment=False)

            batch_size = self.batch_size_var.get()
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

            # Setup training
            class_counts = np.bincount(y_train.astype(int))
            class_weights = torch.FloatTensor([class_counts[1]/class_counts[0], 1.0]).to(self.device)
            criterion = nn.NLLLoss(weight=class_weights)

            learning_rate = self.lr_var.get()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

            epochs = self.epochs_var.get()
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                     steps_per_epoch=len(train_loader), epochs=epochs)

            # Training loop
            self.log("="*60)
            self.log("Starting training loop...")
            self.log("="*60)

            best_val_acc = 0
            train_losses = []
            val_accuracies = []

            for epoch in range(epochs):
                if not self.training_active:
                    break

                # Training
                model.train()
                train_loss = 0

                for batch_idx, (data_batch, target) in enumerate(train_loader):
                    if not self.training_active:
                        break

                    data_batch, target = data_batch.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = model(data_batch)
                    loss = criterion(output, target)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    scheduler.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # Validation
                model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for data_batch, target in val_loader:
                        data_batch, target = data_batch.to(self.device), target.to(self.device)
                        output = model(data_batch)
                        pred = output.argmax(dim=1)
                        val_correct += pred.eq(target).sum().item()
                        val_total += target.size(0)

                val_acc = 100. * val_correct / val_total
                val_accuracies.append(val_acc)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'input_dim': input_dim,
                        'sequence_length': sequence_length,
                        'model_class': model_class.__name__
                    }, self.output_path_var.get())

                # Update UI
                progress = ((epoch + 1) / epochs) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.2f}%")

                if epoch % 5 == 0:
                    self.log(f"Epoch {epoch+1}/{epochs}: Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if not self.training_active:
                self.log("Training was stopped early")
                return

            # Test evaluation
            self.status_var.set("Evaluating on test set...")
            self.log("\n" + "="*60)
            self.log("Evaluating on test set...")

            model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for data_batch, target in test_loader:
                    data_batch, target = data_batch.to(self.device), target.to(self.device)
                    output = model(data_batch)
                    pred = output.argmax(dim=1)
                    test_correct += pred.eq(target).sum().item()
                    test_total += target.size(0)

            test_acc = 100. * test_correct / test_total

            # Display results
            self.log("="*60)
            self.log("TRAINING COMPLETE!")
            self.log("="*60)
            self.log(f"Best Validation Accuracy: {best_val_acc:.2f}%")
            self.log(f"Test Accuracy: {test_acc:.2f}%")
            self.log(f"Model saved to: {self.output_path_var.get()}")
            self.log(f"Total parameters: {total_params:,}")

            # Update results tab
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "TRAINING RESULTS\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")
            self.results_text.insert(tk.END, f"Model: {model_name}\n")
            self.results_text.insert(tk.END, f"Parameters: {total_params:,}\n")
            self.results_text.insert(tk.END, f"Epochs Trained: {epochs}\n")
            self.results_text.insert(tk.END, f"Batch Size: {batch_size}\n")
            self.results_text.insert(tk.END, f"Learning Rate: {learning_rate}\n\n")
            self.results_text.insert(tk.END, f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
            self.results_text.insert(tk.END, f"Final Test Accuracy: {test_acc:.2f}%\n\n")
            self.results_text.insert(tk.END, f"Model saved to:\n{self.output_path_var.get()}\n\n")
            self.results_text.insert(tk.END, "Training Loss History:\n")
            for i, loss in enumerate(train_losses[::10]):
                self.results_text.insert(tk.END, f"  Epoch {i*10}: {loss:.4f}\n")
            self.results_text.config(state=tk.DISABLED)

            self.status_var.set(f"Training complete! Test Acc: {test_acc:.2f}%")

            messagebox.showinfo("Success", f"Training completed!\n\nTest Accuracy: {test_acc:.2f}%\nModel saved to: {self.output_path_var.get()}")

        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Training Error", f"An error occurred during training:\n\n{str(e)}")
            self.status_var.set("Training failed")

        finally:
            self.training_active = False
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = TrainingUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
