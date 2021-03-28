import torch
from pytorch_lightning.core.lightning import LightningModule
from  metrics import Metrics
from losses import GaussianLaplaceMixtureNLL
from distributions import GaussianLaplaceMixtureDistribution

class GaussianLaplaceMixtureNLLModel(LightningModule):
    def __init__(self, input_size, y_scale):
        super().__init__()
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, 100),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(100, 100),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(100, 5))
        self.loss = GaussianLaplaceMixtureNLL()

    def forward(self, x):
        x = self.layers(x)
        mu, logvar, loc, logscale, weight = torch.chunk(x, chunks=2, dim=1)
        var = torch.exp(logvar)
        scale = torch.exp(logscale)
        weight = torch.sigmoid(weight)
        return mu, var, loc, scale, weight

    def training_step(self, batch, batch_idx):
        x, y = batch
        params = self(x)
        l = self.loss(y, *params)
        tensorboard_logs = {"train_loss": l}
        return {"loss": l, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        params  = self(x)
        loss = self.loss(y, *params)
        cpu_params = tuple([ params[i].detach().cpu.flatten() for i in range(len(params))])
        dist = GaussianLaplaceMixtureDistribution(cpu_params)
        metrics = Metrics(dist, y.detach().cpu(), self.y_scale, 5000)
        dic["val_loss"] = loss
        dic["point_calibration_error"] = metrics.point_calibration_error()
        return dic

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        params = self(x)
        dic = {
            "test_loss": self.loss(y, *params),
        }
        cpu_params = tuple([ params[i].detach().cpu.flatten() for i in range(len(params))])
        dist = GaussianLaplaceMixtureDistribution(cpu_params)
        metrics = Metrics(dist, y.detach().cpu(), self.y_scale, 5000)
        dic2 = metrics.get_metrics(decision_making=True)
        dic.update(dic2)
        return dic
