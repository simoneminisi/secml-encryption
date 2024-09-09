from tenseal import Context
from torch import nn
from tenseal.tensors.ckksvector import CKKSVector


class Conv2dLayer:
    """
    A class representing a 2D convolutional layer that can potentially be encrypted.

    This class wraps a PyTorch Conv2d layer and provides functionality
    for both encrypted and non-encrypted forward passes. Note that
    encryption is not currently supported for this layer.
    """

    def __init__(
        self,
        layer: nn.Conv2d,
        input_size: tuple,
        encrypt: bool = False,
        context: Context = None,
    ):
        """
        Initialize the Conv2dLayer.

        Args:
            layer (nn.Conv2d): The PyTorch Conv2d layer to be wrapped.
            input_size (tuple): The size of the input tensor (height, width).
            encrypt (bool): Whether to encrypt the layer weights and biases (not supported).
            context (Context): The encryption context to use if encrypting (not used).

        Raises:
            Exception: If encryption is attempted, as it's not supported for Conv2dLayer.
        """
        self._input_size = input_size
        self._encrypt = encrypt

        if layer != None:
            self._kernel_size = layer.kernel_size
            self._stride = layer.stride

            self._weight = layer.weight.data.view(
                layer.out_channels, self._kernel_size[0], self._kernel_size[1]
            ).tolist()
            self._bias = layer.bias.data.tolist()

        if encrypt:
            raise Exception("Encryption is not supported with Conv2DLayer")

    def __call__(self, x: CKKSVector):
        """
        Perform a forward pass through the convolutional layer.

        This method handles the convolution operation using the im2col method
        for CKKSVector inputs.

        Args:
            x (CKKSVector): The input tensor as a CKKSVector.

        Returns:
            CKKSVector: The output of the convolutional layer.
        """
        stride = self._stride[0]

        out_height = (self._input_size[0] - self._kernel_size[0]) / stride + 1
        out_width = (self._input_size[1] - self._kernel_size[1]) / stride + 1

        # windows number
        windows_nb = int(out_height * out_width)

        enc_channels = []

        for kernel, bias in zip(self._weight, self._bias):
            y = x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)

        # pack all channels into a single flattened vector
        return CKKSVector.pack_vectors(enc_channels)

    def serialize(self):
        """
        Serialize the convolutional layer.

        This method creates a dictionary representation of the layer,
        which can be used for saving or transmitting the layer.

        Returns:
            dict: A dictionary containing the serialized layer data.
        """
        return {
            "weight": self._weight,
            "bias": self._bias,
            "kernel_size": self._kernel_size,
            "input_size": self._input_size,
            "stride": self._stride,
            "type": "Conv2d",
            "encrypt": self._encrypt,
        }
