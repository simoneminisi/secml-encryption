from __future__ import annotations
from tenseal.tensors.ckksvector import CKKSVector
from torch.nn.modules.module import Module
from torch import Tensor


class SqNL(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{SqNL}(x) = x * x

    This activation function squares each element of the input.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SqNL()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor | CKKSVector) -> Tensor | CKKSVector:
        """
        Applies the SqNL (Square Non-Linearity) function element-wise.

        Args:
            input (Tensor | CKKSVector): The input tensor or CKKSVector.

        Returns:
            Tensor | CKKSVector: The output with the same shape as the input, where each element is the square of the corresponding input element.
        """
        return input * input

    def serialize(self):
        """
        Serialize the SqNL activation function.

        This method creates a dictionary representation of the activation function,
        which can be used for saving or transmitting the layer.

        Returns:
            dict: A dictionary containing the serialized activation function data.
        """
        return {"type": "SqNL"}
