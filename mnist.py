import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from collections import defaultdict
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageViewer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.current_idx = 0
        
        # Create a new figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.canvas.manager.set_window_title('MNIST Visualization')
        
        # Connect key event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial plot
        self.update_plot()
        plt.tight_layout()
        
    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        
        # Get image and process it
        image, label = self.dataset[self.current_idx]
        image_np = image.squeeze().numpy()
        
        # Denormalize the image
        image_np = image_np * 0.3081 + 0.1307
        image_np = np.clip(image_np, 0, 1)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().squeeze()
            
        # Plot image
        self.ax1.imshow(image_np, cmap='gray')
        self.ax1.set_title(f'True Label: {label}')
        self.ax1.axis('off')
        
        # Plot probabilities
        probs = probs.squeeze().cpu()
        probs_np = probs.numpy()
        print(probs_np)
        bars = self.ax2.bar(range(10), probs_np)
        predicted = probs_np.argmax()
        
        # Color coding
        for bar in bars:
            bar.set_color('lightgray')
        bars[predicted].set_color('red')
        bars[label].set_color('green')
        
        # Add probability values on top of bars
        for i, prob in enumerate(probs_np):
            self.ax2.text(i, prob, f'{prob:.2f}', ha='center', va='bottom')
        
        self.ax2.set_title(f'Predicted: {predicted} (Confidence: {probs_np[predicted]:.2f})')
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xticks(range(10))
        self.ax2.set_xlabel('Digit')
        self.ax2.set_ylabel('Probability')
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def on_key_press(self, event):
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.dataset)
            self.update_plot()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.dataset)
            self.update_plot()
        elif event.key == 'escape':
            plt.close(self.fig)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EpochStats(BaseModel):
    """Statistics for a single training epoch"""
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    learning_rate: float
    
class TrainingRun(BaseModel):
    """Complete training run information and history"""
    # Training hyperparameters
    learning_rate: float
    batch_size: int
    num_epochs: int
        
    # Training history
    epoch_history: List[EpochStats] = []
    
    # Final metrics
    final_train_loss: float = None
    final_train_accuracy: float = None
    final_test_loss: float = None
    final_test_accuracy: float = None

    def add_epoch_stats(self, 
                       epoch: int,
                       train_loss: float,
                       train_accuracy: float,
                       test_loss: float,
                       test_accuracy: float,
                       learning_rate: float):
        """Add statistics for a single epoch"""
        stats = EpochStats(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            learning_rate=learning_rate
        )

        self.final_train_loss = train_loss
        self.final_train_accuracy = train_accuracy
        self.final_test_loss = test_loss
        self.final_test_accuracy = test_accuracy

        self.epoch_history.append(stats)

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, test_dataset

def evaluate_model(model, test_loader):
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, test_loss / len(test_loader)

def train_model(model, train_loader, test_loader, test_dataset, num_epochs=10, learning_rate=0.001, verbose=False) -> TrainingRun:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    info = TrainingRun(
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        num_epochs=num_epochs
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0 and verbose:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, Accuracy: {100 * correct/total:.2f}%')
                running_loss = 0.0
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        test_acc, test_loss = evaluate_model(model, test_loader)
        
        info.add_epoch_stats(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_accuracy=train_acc,
            test_loss=test_loss,
            test_accuracy=test_acc,
            learning_rate=learning_rate
        )
                
        # Create visualization window
        if verbose:
            viewer = ImageViewer(test_dataset, model)
            plt.show()  # Blocks until the window is closed
        
    return info


if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_EPOCHS = 1
    LR_RANGE = [0.007, 0.01]
    lrs = np.geomspace(*LR_RANGE, 50)
    
    # Dictionary to store results for each learning rate
    results = defaultdict(list)
    
    train_loader, test_loader, test_dataset = get_data_loaders(BATCH_SIZE)
    
    runs: List[TrainingRun] = []

    for lr in lrs:
        model = MNISTNet().to(device)
        run = train_model(model, train_loader, test_loader, test_dataset, NUM_EPOCHS, lr)
        runs.append(run)
        print(f'Finished LR = {lr:.5f}, Test Accuracy = {run.final_test_accuracy:.2f}%')

    # Extract metrics from the existing runs list
    train_accuracies = [run.final_train_accuracy for run in runs]
    train_losses = [run.final_train_loss for run in runs]
    test_accuracies = [run.final_test_accuracy for run in runs]
    test_losses = [run.final_test_loss for run in runs]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot each metric
    # plt.semilogx(lrs, train_accuracies, 'b-o', label='Train Accuracy (%)')
    # plt.semilogx(lrs, test_accuracies, 'g-o', label='Test Accuracy (%)')
    plt.semilogx(lrs, train_losses, 'r-o', label='Train Loss')
    plt.semilogx(lrs, test_losses, 'm-o', label='Test Loss')

    # Customize the plot
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Learning Rate')
    plt.ylabel('Metrics')
    plt.title('Impact of Learning Rate on Model Performance')
    plt.legend()

    # Add a second y-axis for accuracies
    ax2 = plt.gca().twinx()
    ax2.set_ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()
    
    
    