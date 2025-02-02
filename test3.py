import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

# Set device
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple model (e.g., a 2-layer MLP)
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Define the "meta" model to predict weight updates
class MetaUpdater(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MetaUpdater, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, weight_flattened, gradient_flattened):
        x = torch.cat([weight_flattened, gradient_flattened], dim=-1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # Predicts weight update

# Load Fashion-MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = SimpleModel(input_dim=28*28, hidden_dim=128, output_dim=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Collect weight transitions
weight_data = []
grad_data = []
target_updates = []

for epoch in range(20):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.view(batch_X.size(0), -1).to(device), batch_y.to(device)  # Flatten images
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Store weight and gradient information
        weights = torch.cat([p.view(-1) for p in model.parameters()]).detach().clone()
        gradients = torch.cat([p.grad.view(-1) for p in model.parameters()]).detach().clone()
        
        weight_data.append(weights)
        grad_data.append(gradients)
        optimizer.step()
        
        # Store target weight updates
        new_weights = torch.cat([p.view(-1) for p in model.parameters()]).detach().clone()
        target_updates.append(new_weights - weights)

# Convert collected data into tensors
weight_data = torch.stack(weight_data).to(device)
grad_data = torch.stack(grad_data).to(device)
target_updates = torch.stack(target_updates).to(device)

# Train the MetaUpdater model
meta_updater = MetaUpdater(input_dim=weight_data.shape[1] * 2, hidden_dim=100, output_dim=weight_data.shape[1]).to(device)
meta_optim = optim.Adam(meta_updater.parameters(), lr=0.01)
meta_criterion = nn.MSELoss()

meta_dataset = TensorDataset(weight_data, grad_data, target_updates)
meta_loader = DataLoader(meta_dataset, batch_size=32, shuffle=True)

losses = []
for epoch in range(100):
    epoch_loss = 0
    for w, g, t in meta_loader:
        w, g, t = w.to(device), g.to(device), t.to(device)
        meta_optim.zero_grad()
        pred_updates = meta_updater(w, g)
        loss = meta_criterion(pred_updates, t)
        loss.backward()
        meta_optim.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(meta_loader))
    print(f"Epoch {epoch}: Meta Loss = {losses[-1]:.6f}")

print("Meta-updater training complete.")

# Test: Apply meta-learned weight updates
model_test = SimpleModel(input_dim=28*28, hidden_dim=128, output_dim=10).to(device)
model_test.eval()
test_weights = torch.cat([p.view(-1) for p in model_test.parameters()]).to(device)
test_grads = torch.randn_like(test_weights).to(device)  # Fake gradient for testing
predicted_update = meta_updater(test_weights, test_grads)
new_weights = test_weights + predicted_update  # Apply learned update rule
print("Predicted update applied.")

# Evaluate on test dataset
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Accuracy before and after applying predicted updates
before_acc = evaluate_model(model_test, test_loader)
print(f"Test Accuracy before meta-update: {before_acc:.4f}")

# Apply predicted update to model
with torch.no_grad():
    start_idx = 0
    for p in model_test.parameters():
        end_idx = start_idx + p.numel()
        p.copy_((test_weights + predicted_update)[start_idx:end_idx].view(p.shape))
        start_idx = end_idx

after_acc = evaluate_model(model_test, test_loader)
print(f"Test Accuracy after meta-update: {after_acc:.4f}")

# Compare with standard training
standard_model = SimpleModel(input_dim=28*28, hidden_dim=128, output_dim=10).to(device)
standard_optimizer = optim.Adam(standard_model.parameters(), lr=0.01)

for _ in range(1):  # One standard training step
    for images, labels in train_loader:
        images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
        standard_optimizer.zero_grad()
        outputs = standard_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        standard_optimizer.step()
        break  # Just one batch for fair comparison

standard_acc = evaluate_model(standard_model, test_loader)
print(f"Test Accuracy after one step of standard training: {standard_acc:.4f}")
