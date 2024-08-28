import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tenseal as ts
from secmltencryption.models.pytorch.model_wrapper import ModelWrapper

import os
import pickle
import base64

def read_data(file_name: str) -> bytes:
    with open(file_name, 'rb') as f:
        data = f.read()
    return base64.b64decode(data)

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./examples/data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

serialized_model_file = open(os.path.join('./examples/Scenario2/Example1', 'serialized_model.obj'), 'rb') 
serialized_model = pickle.load(serialized_model_file)

public_key = read_data(os.path.join('./examples/Scenario2/Example1', "public.txt"))

wrapped_model = ModelWrapper.deserialize(key=public_key, serialized_model=serialized_model)
encrypted_outputs = []
targets = []

# Testing loop
c = 0
with torch.no_grad():
    for data, target in test_loader:
        print('Sample:', c)
        encrypted_outputs += [wrapped_model(data.view(-1, 28*28).flatten()).serialize()]
        targets += [x for x in target]
        if c > 20:
            break
        c += 1

filehandler = open(os.path.join('./examples/Scenario2/Example1', 'encrypted_outputs.obj'), 'wb') 
pickle.dump(encrypted_outputs, filehandler)

filehandler = open(os.path.join('./examples/Scenario2/Example1', 'targets.obj'), 'wb') 
pickle.dump(targets, filehandler)
