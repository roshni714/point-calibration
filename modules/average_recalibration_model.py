import torch
from sklearn.isotonic import IsotonicRegression
import numpy as np
import math
from metrics import Metrics
from distributions import FlexibleDistribution
RANGE = [-10, 10]
RESOLUTION=8000


class AverageRecalibrationModel:

    def __init__(self, datasets, y_scale):
        self.train_dist, self.y_train, self.val_dist, self.y_val, self.test_dist, self.y_test = datasets 
        self.y_scale = y_scale
        self.n_bins_test = int(math.sqrt(self.y_train.shape[0]))

    def training_step(self):
        train_forecasts = self.train_dist.cdf(self.y_train.flatten())
        sorted_forecasts = torch.sort(train_forecasts)[0].flatten().numpy()
        Y = np.array([(i+1)/(len(sorted_forecasts) +2) for i in range(len(sorted_forecasts))])
        Y = np.insert(np.insert(Y, 0, 0), len(Y)+1, 1)
        sorted_forecasts = np.insert(np.insert(sorted_forecasts, 0, 0), len(sorted_forecasts)+1, 1)
        self.iso_reg = IsotonicRegression().fit(sorted_forecasts.flatten(), Y)

    def testing_step(self):
        cdfs = []
        y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)

        params = self.test_dist.params 
        for i in range(self.test_dist.params[0].shape[0]):
            sub_params = tuple([params[j][[i]] for j in range(len(params))])
            small_test_dist = self.test_dist.__class__(sub_params)
            test_forecast = small_test_dist.cdf(y)
            res = self.iso_reg.predict(test_forecast)
            cdfs.append(res.flatten())
        ranking = torch.tensor(cdfs)
        dist = FlexibleDistribution((y, ranking))
        metrics = Metrics(dist, self.y_test, self.y_scale, discretization=self.n_bins_test)
        dic = metrics.get_metrics(decision_making=True)
        return dic

    def validation_step(self):
        cdfs = []
        y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)

        params = self.val_dist.params 
        for i in range(self.val_dist.params[0].shape[0]):
            sub_params = tuple([params[j][[i]] for j in range(len(params))])
            small_val_dist = self.val_dist.__class__(sub_params)
            test_forecast = small_val_dist.cdf(y)
            res = self.iso_reg.predict(test_forecast)
            cdfs.append(res.flatten())
        ranking = torch.tensor(cdfs)
        dist = FlexibleDistribution((y, ranking))
        metrics = Metrics(dist, self.y_val, self.y_scale, discretization=self.n_bins_test)
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
