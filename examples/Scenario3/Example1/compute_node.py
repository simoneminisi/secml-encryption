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

serialized_model_file = open(os.path.join('./examples/Scenario3/Example1', 'serialized_model.obj'), 'rb') 
serialized_model = pickle.load(serialized_model_file)

public_key = read_data(os.path.join('./examples/Scenario3/Example1', "public.txt"))

filehandler = open(os.path.join('./examples/Scenario3/Example1', 'encrypted_data.obj'), 'rb') 
encrypted_data = pickle.load(filehandler)

wrapped_model = ModelWrapper.deserialize(key=public_key, serialized_model=serialized_model)
encrypted_outputs = []
targets = []

encrypted_data = [ts.ckks_vector_from(wrapped_model._context, x) for x in encrypted_data]

# Testing loop
c = 0
with torch.no_grad():
    for data in encrypted_data:
        print('Sample:', c)
        encrypted_outputs += [wrapped_model(data).serialize()]
        if c > 20:
            break
        c += 1

filehandler = open(os.path.join('./examples/Scenario3/Example1', 'encrypted_outputs.obj'), 'wb') 
pickle.dump(encrypted_outputs, filehandler)
