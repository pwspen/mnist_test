import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# Add at the top with other imports
from matplotlib.widgets import Button

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageViewer:
    def __init__(self, dataset, model, sorted_indices=None):
        self.dataset = dataset
        self.model = model
        self.current_idx = 0
        self.sorted_indices = sorted_indices if sorted_indices is not None else np.arange(len(dataset))
        self.sort_mode = True  # Start in sorted mode
        
        # Create figure and buttons
        self.fig = plt.figure(figsize=(12, 5))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.fig.canvas.manager.set_window_title('MNIST Visualization - Sorted by Confidence')
        
        # Add toggle button
        self.btn_ax = self.fig.add_axes([0.4, 0.02, 0.2, 0.05])
        self.toggle_btn = Button(self.btn_ax, 'Toggle Sort')
        self.toggle_btn.on_clicked(self.toggle_sort)
        
        # Connect key events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.update_plot()
        
    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        
        # Get actual dataset index based on current sort mode
        actual_idx = self.sorted_indices[self.current_idx] if self.sort_mode else self.current_idx
        image, true_label = self.dataset[actual_idx]
        image_np = image.squeeze().numpy()
        
        # Denormalize image
        image_np = image_np * 0.3081 + 0.1307
        image_np = np.clip(image_np, 0, 1)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().squeeze()
        
        # Plot image
        self.ax1.imshow(image_np, cmap='gray')
        self.ax1.set_title(f'True Label: {true_label}\nIndex: {actual_idx}')
        self.ax1.axis('off')
        
        # Plot probabilities
        probs_np = probs.numpy()
        bars = self.ax2.bar(range(10), probs_np)
        predicted = probs_np.argmax()
        correct_prob = probs_np[true_label]
        
        # Color coding
        for bar in bars:
            bar.set_color('lightgray')
        bars[predicted].set_color('red' if predicted != true_label else 'orange')
        bars[true_label].set_color('green')
        
        # Add probability values
        for i, prob in enumerate(probs_np):
            self.ax2.text(i, prob, f'{prob:.2f}', ha='center', va='bottom')
        
        sort_status = "Sorted (Worst First)" if self.sort_mode else "Original Order"
        self.ax2.set_title(f'Predicted: {predicted} | Confidence: {correct_prob:.2f}\n{sort_status}')
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xticks(range(10))
        
        plt.tight_layout()
        self.fig.canvas.draw_idle()
    
    def on_key_press(self, event):
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.dataset)
            self.update_plot()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.dataset)
            self.update_plot()
        elif event.key == 'escape':
            plt.close(self.fig)
    
    def toggle_sort(self, event):
        self.sort_mode = not self.sort_mode
        self.current_idx = 0  # Reset to first item when toggling
        self.update_plot()

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(128*3*3, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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

def compute_sorted_indices(model, dataset):
    model.eval()
    confidences = []
    with torch.no_grad():
        for i in range(len(dataset)):
            image, label = dataset[i]
            output = model(image.unsqueeze(0).to(device))
            prob = torch.softmax(output, dim=1)[0, label].item()
            confidences.append((i, prob))
    # Sort by confidence (ascending - worst predictions first)
    sorted_conf = sorted(confidences, key=lambda x: x[1])
    return [x[0] for x in sorted_conf]

def train_model(model, train_loader, test_loader, test_dataset, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_accuracy = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

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

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, Acc: {100*correct/total:.2f}%')
                running_loss = 0.0

        # Evaluate and sort test set
        test_acc, test_loss = evaluate_model(model, test_loader)
        sorted_indices = compute_sorted_indices(model, test_dataset)

        # Create viewer with sorted indices
        viewer = ImageViewer(test_dataset, model, sorted_indices)
        plt.show()  # This will block until window is closed

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1} Complete\nTest Accuracy: {test_acc:.2f}%')
        print('-'*60)
    
    return history

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, None

if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LR = 0.001

    train_loader, test_loader, test_dataset = get_data_loaders(BATCH_SIZE)
    model = MNISTNet().to(device)
    history = train_model(model, train_loader, test_loader, test_dataset, NUM_EPOCHS, LR)

    model.load_state_dict(torch.load('best_model.pth'))
    final_acc, _ = evaluate_model(model, test_loader)
    print(f'Final Test Accuracy: {final_acc:.2f}%')