from torchinfo import summary
from torch import nn
import tenseal as ts
import os
import base64
from secmltencryption.models.pytorch.linear_layer import LinearLayer
from secmltencryption.models.pytorch.conv2d_layer import Conv2dLayer
from secmltencryption.activation_functions.square import SqNL
import pickle
import torch


class HEWrapper:
    """
    A wrapper class for PyTorch models that supports encryption and serialization.

    This class wraps a PyTorch model and provides functionality for encrypting the model,
    serializing and deserializing it, and performing forward passes with both encrypted
    and non-encrypted data.
    """

    pipeline = []
    supported_classes = ["Linear", "SqNL", "Conv2d"]

    def __init__(
        self,
        model: nn.Module,
        input_size: tuple = (),
        encrypt_model: bool = False,
        bits_scale: int = 26,
        num_matmul: int = 1,
        poly_modulus_degree: int = 8192,
        first_last_bits_scale: int = 31,
    ):
        """
        Initialize the HEWrapper.

        Args:
            model (nn.Module): The PyTorch model to be wrapped.
            input_size (tuple): The input size of the model.
            encrypt_model (bool): Whether to encrypt the model or not.
            bits_scale (int): The bit scale for encryption.
            num_matmul (int): The number of matrix multiplications.
            poly_modulus_degree (int): The polynomial modulus degree for encryption.
            first_last_bits_scale (int): The bit scale for the first and last coefficient modulus.
        """
        self._context = None
        self._input_size = input_size
        self._encrypt_model = encrypt_model
        self._bits_scale = bits_scale
        self._num_matmul = num_matmul
        self._poly_modulus_degree = poly_modulus_degree
        self._first_last_bits_scale = first_last_bits_scale

        if model is None:
            return

        model_summary = summary(model, verbose=0)

        if encrypt_model:
            # Create TenSEAL context
            self._context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=[first_last_bits_scale]
                + [self._bits_scale] * self._num_matmul
                + [first_last_bits_scale],
            )

            self._context.auto_rescale = True
            self._context.auto_relin = True

            # set the scale
            self._context.global_scale = pow(2, bits_scale)

            self._context.generate_galois_keys()

        for x in model_summary.summary_list[1:]:
            if isinstance(x.module, torch.nn.Linear):
                self.pipeline += [
                    LinearLayer(
                        getattr(model, x.var_name),
                        encrypt=encrypt_model,
                        context=self._context,
                    )
                ]
            elif isinstance(x.module, SqNL):
                self.pipeline += [getattr(model, x.var_name)]
            elif isinstance(x.module, torch.nn.Conv2d):
                self.pipeline += [
                    Conv2dLayer(getattr(model, x.var_name), input_size=input_size)
                ]
            else:
                raise Exception(
                    f'Layer of class { x.class_name } not supported. Supported layers are: { str.join(",", self.supported_classes) }'
                )

    def __call__(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x: The input data.

        Returns:
            The output of the model after passing through all layers.
        """
        for layer in self.pipeline:
            x = layer(x)

        return x

    def write_data(self, file_name: str, data: bytes):
        """
        Write binary data to a file in base64 encoding.

        Args:
            file_name (str): The name of the file to write to.
            data (bytes): The binary data to write.
        """
        data = base64.b64encode(data)
        with open(file_name, "wb") as f:
            f.write(data)

    def serialize(self, path):
        """
        Serialize the model and save it to the specified path.

        This method serializes the model's context (if it exists) and the model pipeline.
        It saves the serialized data to files in the specified path.

        Args:
            path (str): The directory path where the serialized model will be saved.
        """
        if self._context != None:
            secret_context = self._context.serialize(save_secret_key=True)
            self.write_data(os.path.join(path, "secret.txt"), secret_context)

            self._context.make_context_public()
            public_context = self._context.serialize()
            self.write_data(os.path.join(path, "public.txt"), public_context)

        serialized_pipeline = [x.serialize() for x in self.pipeline]

        serialized_model = {
            "input_size": self._input_size,
            "encrypt_model": self._encrypt_model,
            "bits_scale": self._bits_scale,
            "num_matmul": self._num_matmul,
            "poly_modulus_degree": self._poly_modulus_degree,
            "serialized_pipeline": serialized_pipeline,
        }

        filehandler = open(os.path.join(path, "serialized_model.obj"), "wb")
        pickle.dump(serialized_model, filehandler)

    @classmethod
    def deserialize(cls, key, serialized_model):
        """
        Deserialize a model from the given serialized data.

        This class method creates a new HEWrapper instance from the serialized data.

        Args:
            key: The encryption key (can be None for non-encrypted models).
            serialized_model: The serialized model data.

        Returns:
            HEWrapper: A new instance of HEWrapper with the deserialized model.
        """
        wrapped_model = HEWrapper(
            None,
            input_size=serialized_model["input_size"],
            encrypt_model=serialized_model["encrypt_model"],
            bits_scale=serialized_model["bits_scale"],
            num_matmul=serialized_model["num_matmul"],
            poly_modulus_degree=serialized_model["poly_modulus_degree"],
        )
        if key == None:
            wrapped_model._context = None
        else:
            wrapped_model._context = ts.context_from(key)

        wrapped_model.pipeline = [
            cls.deserialize_layer(layer, context=wrapped_model._context)
            for layer in serialized_model["serialized_pipeline"]
        ]

        return wrapped_model

    @classmethod
    def deserialize_layer(cls, serialized_layer, context):
        """
        Deserialize a single layer from the given serialized data.

        This class method creates and returns a layer object from the serialized layer data.

        Args:
            serialized_layer: The serialized data for a single layer.
            context: The encryption context (can be None for non-encrypted layers).

        Returns:
            The deserialized layer object.
        """
        if serialized_layer["type"] == "Linear":
            layer = LinearLayer(None, encrypt=serialized_layer["encrypt"])
            if serialized_layer["encrypt"]:
                layer._weight = [
                    ts.ckks_vector_from(context, weight)
                    for weight in serialized_layer["weight"]
                ]
                layer._bias = ts.ckks_vector_from(context, serialized_layer["bias"])
            else:
                layer._weight = serialized_layer["weight"]
                layer._bias = serialized_layer["bias"]

            return layer

        elif serialized_layer["type"] == "SqNL":
            return SqNL()

        elif serialized_layer["type"] == "Conv2d":
            layer = Conv2dLayer(
                None,
                input_size=serialized_layer["input_size"],
                encrypt=serialized_layer["encrypt"],
            )

            layer._weight = serialized_layer["weight"]
            layer._bias = serialized_layer["bias"]
            layer._stride = serialized_layer["stride"]
            layer._kernel_size = serialized_layer["kernel_size"]

            return layer
