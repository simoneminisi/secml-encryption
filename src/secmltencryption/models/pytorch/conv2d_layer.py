from tenseal import Context
from torch import nn
from tenseal.tensors.ckksvector import CKKSVector

class Conv2dLayer:
  def __init__(
      self,
      layer: nn.Conv2d,
      input_size: tuple,
      encrypt: bool = False,
      context: Context = None
  ):
    self._input_size = input_size
    self._encrypt = encrypt

    if layer != None:
      self._kernel_size = layer.kernel_size
      self._stride = layer.stride
    
      self._weight = layer.weight.data.view(
          layer.out_channels, self._kernel_size[0],
          self._kernel_size[1]
      ).tolist()
      self._bias = layer.bias.data.tolist()

    if encrypt:
      raise Exception('Encryption is not supported with Conv2DLayer')

  def __call__(self, x: CKKSVector):
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
    return {
      'weight': self._weight,
      'bias': self._bias,
      'kernel_size': self._kernel_size,
      'input_size': self._input_size,
      'stride': self._stride,
      'type': 'Conv2d',
      'encrypt': self._encrypt
    }