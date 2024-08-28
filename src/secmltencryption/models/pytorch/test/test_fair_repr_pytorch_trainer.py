import unittest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from secml2fair.metrics.regression_fairness_metric import RegressionFairnessMetric
from secml2fair.utils.fair_data_loader import FairDataLoader
from secml2fair.models.pytorch.fair_repr_pytorch_trainer import FairReprPyTorchTrainer
from secml2.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer


class MockFairnessMetric(RegressionFairnessMetric):
    def __call__(self, model, data_loader):
        return torch.tensor(0.5)

    def from_predictions(self, predicted, sa_values):
        return torch.tensor(0.5)


class MockFairLayer(torch.nn.Module):
    def forward(self, x):
        return x


class TestFairReprPyTorchTrainer(unittest.TestCase):
    def setUp(self):
        # Set up necessary data for testing
        self.optimizer = SGD(params=[torch.tensor(1.0)], lr=0.1)
        self.loss = torch.nn.MSELoss()
        self.scheduler = StepLR(self.optimizer, step_size=1)
        self.distinct_sa = torch.tensor([0, 1])
        self.fairness_metric = MockFairnessMetric(self.distinct_sa)
        self.fair_layer = MockFairLayer()
        self.trainer = FairReprPyTorchTrainer(
            optimizer=self.optimizer,
            epochs=3,
            loss=self.loss,
            scheduler=self.scheduler,
            fairness_weight=1.0,
            fairness_metric=self.fairness_metric,
            distinct_sa=self.distinct_sa,
            fair_layer=self.fair_layer,
        )

    def test_initialization(self):
        # Test the initialization of the FairReprPyTorchTrainer class
        self.assertIsInstance(self.trainer, BasePyTorchTrainer)
        self.assertEqual(self.trainer._fairness_weight, 1.0)
        self.assertEqual(self.trainer._fairness_metric, self.fairness_metric)
        self.assertEqual(self.trainer._fair_layer, self.fair_layer)

    def test_train_method(self):
        # Test the train method of the FairReprPyTorchTrainer class
        model = torch.nn.Linear(5, 1)
        data_loader = FairDataLoader(
            [[torch.randn(5), torch.zeros(1), torch.ones(1)[0]] for i in range(20)],
            shuffle=False,
            batch_size=5,
        )
        trained_model = self.trainer.train(model, data_loader)
        self.assertIsInstance(trained_model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
