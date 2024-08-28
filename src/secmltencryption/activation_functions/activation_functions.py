from __future__ import annotations
from tenseal.tensors.ckksvector import CKKSVector
from torch.nn.modules.module import Module
from torch import Tensor


class SqNL(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{SqNL}(x) = x * x


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SqNL()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor | CKKSVector) -> Tensor | CKKSVector:
        return input * input
    
    def serialize(self):
        return {
            'type': 'SqNL'
        }

def sigmoid_approx(x: CKKSVector):
    return 0.5 + 0.197 * x - 0.004 * x.pow(3)