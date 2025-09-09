import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision

class NumpyMNIST(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.data = np.load(x_path)  # (N, 784)
        self.labels = np.load(y_path)  # (N,)
        self.transform = transform
        self.data = self.data.reshape(-1, 28, 28).astype(np.float32) # Reshape to (N, 28, 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.last_conv_length = 64 * 7 * 7
        self.fc1 = nn.Linear(self.last_conv_length, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation = nn.functional.relu

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.last_conv_length)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def get_data_loaders(root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    x_path = os.path.join(root, "X.npy")
    y_path = os.path.join(root, "y.npy")

    if os.path.exists(x_path) and os.path.exists(y_path):
        print(" Using local npy dataset...")
        dataset = NumpyMNIST(x_path, y_path, transform=transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
    else:
        print(" Downloading MNIST dataset via torchvision...")
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader


if __name__ == "__main__":

    root = os.path.join(
        r"D:\Ai\DL\Projects\mnist-sample-20250706T163633Z-1-001",
        "mnist-sample"
    )

    trainloader, testloader = get_data_loaders(root)

    model = MnistCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    ###   Training loop  ####

    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # inputs shape:  [batch_size, channels, 28, 28] --> 64 MNIST images in a batch, 1 channel, 28x28 pixels
        # labels shape:  [batch_size] (the 64 correct digit labels).
        for inputs, labels in trainloader:
            optimizer.zero_grad() # must reset old gradients to zero , because by default gradients are accumulated in PyTorch
            outputs = model(inputs) # Output shape: [64, 10] (10 classes for each image in 64 images) -> in Each row will select max prediction digit in the 10 classes
            loss = criterion(outputs, labels) # CrossEntropyLoss --> Applies softmax + negative log likelihood loss ( -log(p) ) 
            loss.backward()
            optimizer.step() # Applies the weight updates using the gradients.
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(trainloader):.4f}")


    ###   Evaluation on test set  ###

    correct, total = 0, 0
    model.eval() # Put the CNN into evaluation mode --> because Layers like Dropout && BatchNorm behave differently in training && testing.
    with torch.no_grad(): # Disable gradient calculation for inference to save memory and computations.

        # inputs shape : (batch_size, 1, 28, 28) --> batch_size = 5
        # labels: the true digit values (0–9).
        for inputs, labels in testloader:
            outputs = model(inputs) # outputs shape: (batch_size, 10) → for each image, we get 10 scores (logits)
            _, predicted = torch.max(outputs.data, 1) # select the highest score(from the 10 classes) for each image along dimension 1 .
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f" Accuracy on test set: {100 * correct / total:.2f}%")
