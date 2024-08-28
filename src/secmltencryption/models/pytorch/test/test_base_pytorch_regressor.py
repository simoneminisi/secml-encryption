import unittest
import torch
from torch.utils.data import DataLoader
from secml2fair.models.pytorch.base_pytorch_regressor import BasePytorchRegressor
from secml2.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from secml2.models.base_model import BaseModel
from torch.optim import SGD


class MockModel(BaseModel):
    def __init__(self):
        self._n_calls = 0

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self._n_calls == 0:
            self._n_calls += 1
        return torch.tensor(
            [1 for _ in range(45)]
            + [0 for _ in range(45)]
            + [1 for _ in range(3)]
            + [0 for _ in range(7)]
            + [1 for _ in range(5)]
            + [0 for _ in range(5)]
            + [1 for _ in range(9)]
            + [0 for _ in range(81)]
        )

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        return super()._decision_function(x)

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        return super().gradient(x, y)

    def train(self, data_loader: DataLoader):
        return super().train(data_loader)


class MockTrainer(BasePyTorchTrainer):
    def train(self, model, data_loader):
        return 42


class LinearRegressor(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearRegressor, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


class TestBasePytorchRegressor(unittest.TestCase):
    def setUp(self):
        # Set up necessary data for testing
        self.mock_model = LinearRegressor(5)
        self.mock_model.to("cpu")
        optimizer = SGD(self.mock_model.parameters(), lr=0.01)
        self.mock_trainer = MockTrainer(optimizer)
        self.data_loader = DataLoader(torch.zeros((10, 5)), batch_size=2)
        self.reggressor = BasePytorchRegressor(
            self.mock_model, trainer=self.mock_trainer
        )

    def test_initialization(self):
        # Test the initialization of the BasePytorchRegressor class
        self.assertIsInstance(self.reggressor, BasePytorchRegressor)
        self.assertEqual(self.reggressor.model, self.mock_model)

    def test_get_device(self):
        # Test the get_device method of the BasePytorchRegressor class
        device = self.reggressor.get_device()
        self.assertEqual(device, next(self.mock_model.parameters()).device)

    def test_predict(self):
        # Test the predict method of the BasePytorchRegressor class
        x = torch.zeros((5, 5))
        result = self.reggressor.predict(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_decision_function(self):
        # Test the _decision_function method of the BasePytorchRegressor class
        x = torch.zeros((5, 5))
        result = self.reggressor._decision_function(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_train(self):
        # Test the train method of the BasePytorchRegressor class
        result = self.reggressor.train(self.data_loader)
        self.assertEqual(result, 42)


if __name__ == "__main__":
    unittest.main()
