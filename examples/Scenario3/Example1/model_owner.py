import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tenseal as ts
from secmltencryption.models.pytorch.model_wrapper import ModelWrapper
from secmltencryption.activation_functions.activation_functions import SqNL
import os
import pickle

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.act1 = SqNL()
        self.fc2 = nn.Linear(10, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input tensor
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./examples/data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='./examples/data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
wrapped_model = ModelWrapper(model, encrypt_model=True, num_matmul=8, poly_modulus_degree=16384, bits_scale=26, first_last_bits_scale=31)
encrypted_data = []
targets = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 28*28).tolist()
        encrypted_data += [ts.ckks_vector(wrapped_model._context, x).serialize() for x in data]
        targets += [x for x in target]
        break
        
filehandler = open(os.path.join('./examples/Scenario3/Example1', 'encrypted_data.obj'), 'wb') 
pickle.dump(encrypted_data, filehandler)

filehandler = open(os.path.join('./examples/Scenario3/Example1', 'targets.obj'), 'wb') 
pickle.dump(targets, filehandler)

wrapped_model.serialize('./examples/Scenario3/Example1')
