import torch
import math
from metrics import Metrics



class NoRecalibrationModel:

    def __init__(self, datasets, y_scale):
        self.train_dist, self.y_train, self.val_dist, self.y_val, self.test_dist, self.y_test = datasets 
        self.y_scale = y_scale
        self.n_bins_test = int(math.sqrt(self.y_train.shape[0]))

    def testing_step(self):
        metrics = Metrics(self.test_dist, self.y_test, self.y_scale, discretization=self.n_bins_test)
        dic = metrics.get_metrics(decision_making=True)
        return dic


    def validation_step(self):
        metrics = Metrics(self.val_dist, self.y_val, self.y_scale, discretization=self.n_bins_test)
        dic = metrics.get_metrics(decision_making=True)
        setattr(self, "val_point_calibration_error", dic["point_calibration_error"].item())
        setattr(self, "val_true_vs_pred_loss", dic["true_vs_pred_loss"].item())
        return dic

    def test_epoch_end(self, outputs):
        for key in outputs[0]:
            if key  not in ["all_err", "all_loss", "all_y0", "all_c"]:
                cal = torch.stack([x[key] for x in outputs]).mean()
                setattr(self, key, float(cal))
            else:
                setattr(self, key, outputs[0][key])

