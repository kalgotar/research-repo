import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

filepath = r'\\training_binaryclass_data.csv'
# Read chroma features and labels from CSV
df = pd.read_csv(filepath)  # CSV file path

# Extract features and labels
features = df.iloc[:, :-1].values  # All columns except the last one
labels = df.iloc[:, -1].values  # The  last column

# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.long)

# Create edge index for a fully connected graph
num_nodes = features.shape[0]
edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()

# Create PyTorch Geometric data object
data = Data(x=features, edge_index=edge_index, y=labels)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and loss function
model = GNN(in_channels=12, out_channels=2)  # binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Create DataLoader
loader = DataLoader([data], batch_size=1, shuffle=True)

# Initialize lists to store loss and accuracy
losses = []
accuracies = []

# Training loop
model.train()
for epoch in range(100):  # Number of epochs
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
    
    # Store the loss
    losses.append(loss.item())
    
    # Evaluate the model
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = (pred == data.y).sum().item()
    accuracy = correct / num_nodes
    accuracies.append(accuracy)
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}')
    model.train()  # Set the model back to training mode

# Plot the loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss - Drone vs. Noise')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy - Drone vs. Noise')
plt.legend()

plt.show()

# Extract the feature embeddings from the model
model.eval()
with torch.no_grad():
    embeddings = model.conv1(data.x, data.edge_index)

# Use t-SNE to reduce the dimensionality to 2D
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings.numpy())

# Plot the 2D embeddings
plt.figure(figsize=(8, 8))
for label in range(2):  # binary classification
    indices = (data.y.numpy() == label)
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=f'Class {label}')

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Feature Embeddings')
plt.legend()
plt.show()

# drone vs background using gnn