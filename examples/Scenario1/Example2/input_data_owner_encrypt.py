import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tenseal as ts

import os
import pickle
import base64

def write_data(file_name: str, data: bytes):
    data = base64.b64encode(data)
    with open(file_name, 'wb') as f: 
        f.write(data)

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./examples/data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# Create TenSEAL context
# controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# set the scale
context.global_scale = pow(2, bits_scale)

context.generate_galois_keys()

encrypted_data = []
targets = []

kernel_shape = (7, 7)
stride = 3

c = 0

with torch.no_grad():
    for data, target in test_loader:
        enc, _ = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )

        encrypted_data += [enc.serialize()]

        targets += [x for x in target]

        c += 1

        if c >= 100:
            break
        
filehandler = open(os.path.join('./examples/Scenario1/Example2', 'encrypted_data.obj'), 'wb') 
pickle.dump(encrypted_data, filehandler)

filehandler = open(os.path.join('./examples/Scenario1/Example2', 'targets.obj'), 'wb') 
pickle.dump(targets, filehandler)

secret_context = context.serialize(save_secret_key = True)
write_data(os.path.join('./examples/Scenario1/Example2', 'secret.txt'), secret_context)
  
context.make_context_public()
public_context = context.serialize()
write_data(os.path.join('./examples/Scenario1/Example2', 'public.txt'), public_context)

print(f'Encrypted {len(targets)} images.')