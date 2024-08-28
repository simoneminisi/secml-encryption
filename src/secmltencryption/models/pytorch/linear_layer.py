from tenseal import Context
from torch import nn
import tenseal as ts
from tenseal.tensors.ckksvector import CKKSVector

class LinearLayer:
  def __init__(
      self,
      layer: nn.Linear,
      encrypt: bool = False,
      context: Context = None
  ):
    self._encrypt = encrypt
    
    if layer == None:
      return
    
    if encrypt:
      self._weight = [ts.ckks_vector(context, column) for column in layer.weight.detach().numpy()]
      self._bias = ts.ckks_vector(context, layer.bias.detach().numpy())
    else:
      self._weight = layer.weight
      self._bias = layer.bias

  def __call__(self, x):
    if self._encrypt:
      tmp = []
      for weight in self._weight:
        tmp += [weight.dot(x)]

      return CKKSVector.pack_vectors(tmp) + self._bias
    else:
      return x.mm(self._weight.T) + self._bias
    
  def serialize(self):
    return {
      'weight': self._weight if not self._encrypt else [weight.serialize() for weight in self._weight],
      'bias': self._bias if not self._encrypt else self._bias.serialize(),
      'type': 'Linear',
      'encrypt': self._encrypt
    }