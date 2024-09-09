from tenseal import Context
from torch import nn
import tenseal as ts
from tenseal.tensors.ckksvector import CKKSVector


class LinearLayer:
    """
    A class representing a linear layer that can be encrypted.

    This class wraps a PyTorch linear layer and provides functionality
    for both encrypted and non-encrypted forward passes.
    """

    def __init__(
        self, layer: nn.Linear, encrypt: bool = False, context: Context = None
    ):
        """
        Initialize the LinearLayer.

        Args:
            layer (nn.Linear): The PyTorch linear layer to be wrapped.
            encrypt (bool): Whether to encrypt the layer weights and biases.
            context (Context): The encryption context to use if encrypting.
        """
        self._encrypt = encrypt

        if layer == None:
            return

        if encrypt:
            self._weight = [
                ts.ckks_vector(context, column)
                for column in layer.weight.detach().numpy()
            ]
            self._bias = ts.ckks_vector(context, layer.bias.detach().numpy())
        else:
            self._weight = layer.weight
            self._bias = layer.bias

    def __call__(self, x):
        """
        Perform a forward pass through the linear layer.

        This method handles both encrypted and non-encrypted inputs.

        Args:
            x: The input tensor or encrypted vector.

        Returns:
            The output of the linear layer.
        """
        if self._encrypt:
            tmp = []
            for weight in self._weight:
                tmp += [weight.dot(x)]

            return CKKSVector.pack_vectors(tmp) + self._bias
        else:
            return x.mm(self._weight.T) + self._bias

    def serialize(self):
        """
        Serialize the linear layer.

        This method creates a dictionary representation of the layer,
        which can be used for saving or transmitting the layer.

        Returns:
            dict: A dictionary containing the serialized layer data.
        """
        return {
            "weight": (
                self._weight
                if not self._encrypt
                else [weight.serialize() for weight in self._weight]
            ),
            "bias": self._bias if not self._encrypt else self._bias.serialize(),
            "type": "Linear",
            "encrypt": self._encrypt,
        }
