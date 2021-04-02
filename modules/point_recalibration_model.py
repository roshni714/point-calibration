import torch
from metrics import Metrics
from pytorch_lightning.core.lightning import LightningModule
from sigmoid import SigmoidFlowND
from composition import CompositionDist
from losses import PointCalibrationLoss
import torch
import torch.nn as nn

class PointRecalibrationModel(LightningModule):

    def __init__(self, datasets, y_scale, n_in=3, n_layers=1, n_dim=100, n_bins=20):
        super().__init__()
        self.y_scale = y_scale
        self.train_dist, self.y_train, self.val_dist, self.y_val, self.test_dist, self.y_test = datasets 
        self.loss = PointCalibrationLoss(discretization=n_bins)
        self.sigmoid_flow = SigmoidFlowND(n_in=n_in, num_layers=n_layers, n_dim=n_dim)

    def training_step(self, batch, batch_idx):
        comp = CompositionDist(self.sigmoid_flow, self.train_dist.to(self.device))
        l = self.loss(self.y_train.to(self.device), comp)
        tensorboard_logs = {"train_loss": l}
        return {"loss": l, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        comp = CompositionDist(self.sigmoid_flow, self.val_dist.to(self.device))
        l = self.loss(self.y_val.to(self.device), comp)
        metrics = Metrics(comp, self.y_val.to(self.device), self.y_scale)
        pce = metrics.point_calibration_error_uniform_mass()
        dic = {}
        dic["point_calibration_error_uniform_mass"] = pce
        dic["val_loss"] = l
        return dic

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        comp = CompositionDist(self.sigmoid_flow, self.test_dist.to(self.device))
        l = self.loss(self.y_test.to(self.device), comp)
        metrics = Metrics(comp, self.y_test.to(self.device), self.y_scale)
        dic = {
            "test_loss": self.loss(self.y_test.to(self.device), comp),
        }
        dic2 = metrics.get_metrics(decision_making=True)
        dic.update(dic2)
        return dic

    def backward(self, loss, optimizer, opt_idx):  
        # do a custom way of backward  
        loss.backward(retain_graph=True)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            if key  not in ["all_err", "all_loss", "all_y0", "all_c"]:
                cal = torch.stack([x[key] for x in outputs]).mean()
                tensorboard_logs[key] = cal
                setattr(self, key, float(cal))
            else:
                setattr(self, key, outputs[0][key])
        return {"test_loss": avg_loss, "log": tensorboard_logs}

