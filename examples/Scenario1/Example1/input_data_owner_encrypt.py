import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tenseal as ts

import os
import pickle
import base64


def write_data(file_name: str, data: bytes):
    data = base64.b64encode(data)
    with open(file_name, "wb") as f:
        f.write(data)


# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
test_dataset = datasets.MNIST(
    root="./examples/data", train=False, download=True, transform=transform
)
test_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=False)

# Create TenSEAL context
# controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, 31],
)

# set the scale
context.global_scale = pow(2, bits_scale)

context.generate_galois_keys()

encrypted_data = []
targets = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 28 * 28).tolist()
        encrypted_data += [ts.ckks_vector(context, x).serialize() for x in data]
        targets += [x for x in target]
        break

filehandler = open(
    os.path.join("./examples/Scenario1/Example1", "encrypted_data.obj"), "wb"
)
pickle.dump(encrypted_data, filehandler)

filehandler = open(os.path.join("./examples/Scenario1/Example1", "targets.obj"), "wb")
pickle.dump(targets, filehandler)

secret_context = context.serialize(save_secret_key=True)
write_data(os.path.join("./examples/Scenario1/Example1", "secret.txt"), secret_context)

context.make_context_public()
public_context = context.serialize()
write_data(os.path.join("./examples/Scenario1/Example1", "public.txt"), public_context)
