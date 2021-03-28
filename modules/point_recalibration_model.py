import torch
from composition import CompositionDist
from losses import PointCalibrationLoss

class PointRecalibrationModel(LightningModule):

    def __init__(self, datasets, n_in=3, num_layers=1, n_dim=100, n_bins=20):
        self.y_train, self.train_dist, self.y_val, self.val_dist, self.y_test, self.test_dist = datasets 
        self.sigmoid_flow = SigmoidFlowND(n_in=n_in, num_layers=n_layers, n_dim=n_dim)
        self.loss = PointCalibrationLoss(discretization=n_bins)

    def training_step(self, batch, batch_idx):
        comp = CompositionDist(self.sigmoid_flow, self.train_dist)
        l = self.loss(self.y_train.to(device).flatten(), comp)
        tensorboard_logs = {"train_loss": l}
        return {"loss": l, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def validation_step(self):
        comp = CompositionDist(self.sigmoid_flow, self.val_dist)
        l = self.loss(self.y_val.to(device).flatten(), comp)
        metrics = Metrics(comp, y_val.detach().cpu(), self.y_scale, 5000)
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

    def test_step(self):
        comp = CompositionDist(self.sigmoid_flow, self.test_dist)
        l = self.loss(self.y_test.to(device).flatten(), comp)
        metrics = Metrics(comp, y_test.detach().cpu(), self.y_scale, 5000)
        dic = {
            "test_loss": self.loss(y, mu, logvar),
        }
        metrics = Metrics(comp, y_test.detach().cpu(), self.y_scale, 5000)
        dic2 = metrics.get_metrics(decision_making=True)
        dic.update(dic2)
        return dic
