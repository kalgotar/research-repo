###### CNN implementation
from torchvision import datasets, transforms,models
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch
import os
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, good_dirs, bad_dirs, transform=None, category='all'):
        self.transform = transform
        
        # Collect all images from the good directories
        good_imgs = []
        for good_dir in good_dirs:
            good_imgs.extend([os.path.join(good_dir, img) for img in os.listdir(good_dir) if img.endswith(('png', 'jpg', 'jpeg'))])
        
        # Collect all images from the bad directories
        bad_imgs = []
        for bad_dir in bad_dirs:
            bad_imgs.extend([os.path.join(bad_dir, img) for img in os.listdir(bad_dir) if img.endswith(('png', 'jpg', 'jpeg'))])
        
        # Label the images
        self.all_imgs = [(img, 0) for img in good_imgs] + [(img, 1) for img in bad_imgs]
        
        # Filter images based on category
        if category == 'good':
            self.all_imgs = [(img, 0) for img in good_imgs]
        elif category == 'bad':
            self.all_imgs = [(img, 1) for img in bad_imgs]
    
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, idx):
        img_path, label = self.all_imgs[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
# Step 1: Define an initial transform (resize and convert to tensor, no normalization yet)
initial_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to a consistent size (e.g., 224x224)
    transforms.ToTensor()           # Convert images to tensor
])

# Step 2: Specify the paths to the different directories
good_dirs = [r'\\FT2D_Drone', r'\\HPSS_Drone', r'\\RepetSim_Drone']
bad_dirs = [r'\\FT2D_Background', r'\\HPSS_Background', r'\\RepetSim_Background']


# # Step 3: Load datasets from each directory using the initial transform
good_datasets = CustomImageDataset(good_dirs=good_dirs, bad_dirs=bad_dirs, transform=initial_transform, category='good')
bad_datasets = CustomImageDataset(good_dirs=good_dirs, bad_dirs=bad_dirs, transform=initial_transform, category='bad')

# # Step 4: Combine the datasets using ConcatDataset
combined_good_dataset = ConcatDataset([good_datasets])
combined_bad_dataset = ConcatDataset([bad_datasets])

# Combine both the 'good' and 'bad' datasets into one
final_dataset = ConcatDataset([good_datasets, bad_datasets])

# # Step 5: Create a DataLoader to iterate over the combined dataset (for calculating mean and std)
loader = DataLoader(final_dataset, batch_size=32, shuffle=False)

# # Step 6: Function to calculate mean and standard deviation of the dataset
def calculate_mean_std(loader):
    # Initialize accumulators
    channel_sum = torch.zeros(3)
    channel_sum_squared = torch.zeros(3)
    num_batches = 0

    # Iterate over the DataLoader
    for data in loader:
        inputs, _ = data
        # Accumulate sum and sum of squares
        channel_sum += inputs.sum(dim=[0, 2, 3])
        channel_sum_squared += (inputs ** 2).sum(dim=[0, 2, 3])
        num_batches += inputs.size(0) * inputs.size(2) * inputs.size(3)

    # Calculate mean and std
    mean = channel_sum / num_batches
    std = (channel_sum_squared / num_batches - mean ** 2).sqrt()

    print(f'Mean: {mean}')
    print(f'Std: {std}')

    return mean, std

# Step 7: Calculate the mean and std for the dataset
mean, std = calculate_mean_std(loader)
print(f"Calculated Mean: {mean}, Calculated Std: {std}")

# # Step 8: Define the final transform including normalization with the calculated mean and std
final_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the desired size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[mean[0].item(), mean[1].item(), mean[2].item()],
                         std=[std[0].item(), std[1].item(), std[2].item()])  # Apply the calculated mean and std
])

# # Step 8: Load the datasets again with the updated transform (with normalization)
good_datasets = CustomImageDataset(good_dirs=good_dirs, bad_dirs=bad_dirs, transform=initial_transform, category='good')
bad_datasets = CustomImageDataset(good_dirs=good_dirs, bad_dirs=bad_dirs, transform=initial_transform, category='bad')

# # Step 9: Combine the datasets using ConcatDataset
combined_good_dataset = ConcatDataset([good_datasets])
combined_bad_dataset = ConcatDataset([bad_datasets])

# Combine both the 'good' and 'bad' datasets into one
final_dataset = ConcatDataset([good_datasets, bad_datasets])

# Step 10: Split the normalized dataset into train, validation, and test sets
train_size = int(0.7 * len(final_dataset))
valid_size = int(0.15 * len(final_dataset))
test_size = len(final_dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(final_dataset, [train_size, valid_size, test_size])

# Step 11: Create DataLoaders for the train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 12: Define a CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleCNN()

# Step 13: Set up the training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train(model, criterion, optimizer, train_loader, val_loader, epochs):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_losses.append(val_loss/len(val_loader))
        print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')
    return train_losses, val_losses

# Step 14: Train the model
train_losses, val_losses = train(model, criterion, optimizer, train_loader, valid_loader, epochs=100)

# Step 15: Test the model
def test(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients for testing
        for images, labels in test_loader:
            outputs = model(images)  # the model outputs
            _, predicted = torch.max(outputs.data, 1)  # the predicted classes
            total += labels.size(0)  # Increment the total count
            correct += (predicted == labels).sum().item()  # Increment the correct count

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# Used trained model and a test_loader for testing
test(model, test_loader)

# Step 16: Plot Training, Validation Loss vs. Epoch
def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(15,5))
    plt.plot(train_losses , label='Training Loss')
    plt.plot(val_losses , label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()

plot_loss_curve(train_losses, val_losses)
