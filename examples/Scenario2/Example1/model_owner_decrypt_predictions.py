import tenseal as ts
import torch

import pickle
import base64
import os


def read_data(file_name: str) -> bytes:
    with open(file_name, "rb") as f:
        data = f.read()
    return base64.b64decode(data)


file_encrypted_output = open(
    os.path.join("./examples/Scenario2/Example1", "encrypted_outputs.obj"), "rb"
)
encrypted_outputs = pickle.load(file_encrypted_output)

file_target = open(os.path.join("./examples/Scenario2/Example1", "targets.obj"), "rb")
targets = pickle.load(file_target)

correct = 0
total = 0

context = ts.context_from(
    read_data(os.path.join("./examples/Scenario2/Example1", "secret.txt"))
)

for encrypted_output, target in zip(encrypted_outputs, targets):
    encrypted_output = ts.ckks_vector_from(context, encrypted_output)
    decrypted_output = encrypted_output.decrypt()
    decrypted_output = torch.tensor(decrypted_output)

    # Evaluate the decrypted outputs
    _, predicted = torch.max(decrypted_output, dim=0)

    total += 1
    correct += (predicted == target).sum().item()

print(
    f"Test Accuracy of the model on the {total} test images with TenSEAL: {100 * correct / total:.2f}%"
)
