import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from secmltencryption.models.pytorch.he_wrapper import HEWrapper
from secmltencryption.activation_functions.square import SqNL
import tenseal as ts

import pickle
import os
import base64


def read_data(file_name: str) -> bytes:
    with open(file_name, "rb") as f:
        data = f.read()
    return base64.b64decode(data)


# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.act1 = SqNL()
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = x.view(-1, 256)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST(
    root="./examples/data", train=True, download=True, transform=transform
)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

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
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

print("Training finished!")

wrapped_model = HEWrapper(model, input_size=(28, 28))

wrapped_model.serialize("./examples/Scenario1/Example2")

serialized_model_file = open(
    os.path.join("./examples/Scenario1/Example2", "serialized_model.obj"), "rb"
)
serialized_model = pickle.load(serialized_model_file)

wrapped_model = HEWrapper.deserialize(key=None, serialized_model=serialized_model)

# Testing loop with TenSEAL
model.eval()
filehandler = open(
    os.path.join("./examples/Scenario1/Example2", "encrypted_data.obj"), "rb"
)
encrypted_data = pickle.load(filehandler)

context = ts.context_from(
    read_data(os.path.join("./examples/Scenario1/Example2", "public.txt"))
)

encrypted_data = [ts.ckks_vector_from(context, x) for x in encrypted_data]

encrypted_output = []

c = 0
with torch.no_grad():
    for enc_x in encrypted_data:
        outputs = wrapped_model(enc_x)
        encrypted_output.append(outputs.serialize())
        c += 1

filehandler = open(
    os.path.join("./examples/Scenario1/Example2", "encrypted_output.obj"), "wb"
)
pickle.dump(encrypted_output, filehandler)
