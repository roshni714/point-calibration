import torch
import numpy as np
import math
from decision_making import simulate_decision_making, DecisionMaker
from scipy.integrate import simps
import torch.distributions as D
torch.manual_seed(0)

class Metrics:
    def __init__(self, dist, y, y_scale):
        self.y = y.flatten()
        self.ft_yt = dist.cdf(self.y).detach().cpu()
        self.dist = dist

        self.y_scale = y_scale
        
        sampled_y0 = torch.linspace(torch.min(y), torch.max(y), 50)
        sampled_alpha = torch.linspace(0.05, 0.95, 50)
        self.decision_makers = []
        for i in range(len(sampled_alpha)):
            for j in range(len(sampled_y0)):
                self.decision_makers.append(DecisionMaker(sampled_alpha[i], sampled_y0[j], self.dist))

    def ece(self):
         ft_yt = torch.sort(self.ft_yt)[0]
         bins= np.linspace(0, 1, ft_yt.shape[0])
         res = simps(np.abs(ft_yt - bins), bins)
         return torch.tensor(res)

    def sharpness(self):
        return self.dist.dist_std.mean().detach().cpu()
 
    def point_calibration_error(self, min_bin=10, discretization=20):
        y_sorted = torch.sort(self.y.flatten())[0]
        n_bins = discretization
        n_y_bins = 50
        sampled_y0 = torch.FloatTensor(n_y_bins).uniform_(torch.min(self.y), torch.max(self.y))
        sampled_alphas = torch.linspace(0, 1, n_bins)
        right_alphas = sampled_alphas[1:].reshape(-1, 1).flatten()
        left_alphas = sampled_alphas[:-1].reshape(-1, 1).flatten()
 
        cdf_vals = self.ft_yt.flatten()
        total_err = 0.
        count = 0
        for k in range(n_y_bins):
            if "Composition" in self.dist.__class__.__name__:
                threshold_vals = self.dist.cdf( sampled_y0[k].to(self.y.get_device())).detach().cpu().reshape(1, -1, 1)
            else:
                threshold_vals = self.dist.cdf( sampled_y0[k].view(-1, 1)).reshape(1, -1, 1)
            selected_indices = (threshold_vals < right_alphas) & (threshold_vals  >= left_alphas) # 2 x 2100 x 199
            num_selected = selected_indices.type(torch.int).sum(dim=1) # 2 x 199
            indices = torch.where(num_selected >= min_bin)
            errs = torch.zeros(len(indices[0]))
            for x in range(len(indices[0])):
                i = indices[0][x]
                j = indices[1][x]
                selected_cdf = cdf_vals[selected_indices[i, :, j]]
                diff_from_uniform = torch.abs(torch.sort(selected_cdf)[0] -torch.linspace(0.0, 1.0, selected_cdf.shape[0])).mean() #*selected_cdf.shape[0]/cdf_vals.shape[0]
                errs[x] = diff_from_uniform
                count += 1
            total_err += errs.sum()
        return total_err/count


    def point_calibration_error_uniform_mass(self, discretization=20):
        n_bins = discretization
        n_y_bins = 50 
        thresholds = torch.linspace(self.y.min(), self.y.max(), n_y_bins)
        count = 0
        pce_mean = 0
        bin_size = int(self.y.shape[0]/discretization)
        cdf_vals = self.ft_yt.flatten()
        for i in range(n_y_bins):
            if "Composition" in self.dist.__class__.__name__:
                threshold_vals = self.dist.cdf(thresholds[[i]].to(self.y.get_device()) ).flatten()
            else:
                threshold_vals = self.dist.cdf(thresholds[i].view(-1, 1)).flatten()
       
            sorted_thresholds, sorted_indices = torch.sort(threshold_vals)
            for x in range(discretization):
                if x != discretization -1:
                    selected_indices = sorted_indices[x * bin_size: (x+1) * bin_size]
                else:
                    selected_indices = sorted_indices[(x) * bin_size:]
                selected_cdf = cdf_vals[selected_indices]
                pce_mean +=  torch.abs(torch.sort(selected_cdf)[0] -torch.linspace(0.0, 1.0, selected_cdf.shape[0])).mean() #*selected_cdf.shape[0]/cdf_vals.shape[0]
                count += 1
        return pce_mean/count


    def rmse(self):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        rmse = torch.sqrt(mse_loss(self.y, self.dist.dist_mean())) * self.y_scale
        return rmse

    def decision_loss(self):
        loss = simulate_decision_making(self.decision_makers, self.dist, self.y.flatten()) # self.point_recal, self.point_recal_params)
        return loss 

    def get_metrics(self, decision_making=False):
        uniform_mass_pce = self.point_calibration_error_uniform_mass()
        if decision_making:
            decision_err, decision_loss, all_err, all_loss, all_y0, all_c = self.decision_loss()
        else:
            decision_loss = torch.tensor(0.)
            decision_err=torch.tensor(0.)
            all_err = torch.tensor([0.])
            all_loss = torch.tensor([0.])
            all_y0 = torch.tensor([0.])
            all_c = torch.tensor([0.])
        return {"ece": self.ece(),
                "point_calibration_error_uniform_mass": uniform_mass_pce, 
                "point_calibration_error": self.point_calibration_error(),
#                "rmse": self.rmse(),
                "true_vs_pred_loss": decision_err, 
                "decision_loss": decision_loss,
                "all_err": all_err, 
                "all_loss": all_loss, 
                "all_y0": all_y0, 
                "all_c": all_c}

