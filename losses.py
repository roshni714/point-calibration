import torch
import torch.distributions as D
import math

class GaussianNLL:
    def __init__(self):
        self.name = "gaussian_nll"

    def __call__(self, y, mu, var):
        nll = 0
        sigma = torch.sqrt(var)
        comp = D.Normal(mu, sigma)
        log_prob = comp.log_prob(y)
        nll += -log_prob
        return torch.mean(nll)

class GaussianLaplaceMixtureNLL:
    def __init__(self):
        self.name = "gaussian_laplace_mixture_nll"

    def __call__(self, y, mu, var, loc, scale, weight):
        gaussian_likelihood = (1/torch.sqrt(2 * math.pi * var)) * torch.exp(- 0.5 * torch.pow(y-mu, 2)/var) * weight 
        laplace_likelihood = (1/(2 * scale)) * torch.exp(-torch.abs(y - loc)/scale) * (1 - weight)
        likelihood = laplace_likelihood + gaussian_likelihood
        likelihood = likelihood.clamp(min=1e-20)
        nll = - torch.log(likelihood)
        return torch.mean(nll)
 
class PointCalibrationLoss:
    def __init__(self, discretization):
        self.name = "pointwise_calibration_loss"
        self.discretization = discretization

    def __call__(self, y, dist):
        n_bins = self.discretization
        n_y_bins = 50
#        with torch.no_grad():
        with torch.no_grad():
            labels_sorted = torch.sort(y.flatten())[0]
            sampled_index = ((torch.rand(n_y_bins) * 0.8 + 0.1) * y.shape[0]).type(torch.long)
            thresholds = labels_sorted[sampled_index]
        vals = []
        for k in range(n_y_bins):
            sub = dist.cdf(thresholds[k]).unsqueeze(dim=0)
            vals.append(sub)
        threshold_vals = torch.cat(vals, dim=0)
        sorted_thresholds, sorted_indices = torch.sort(threshold_vals, dim=1)
        total = 0
        count = 0
        indices= (torch.linspace(0, 1, n_bins, device=y.get_device()) * (y.shape[0]-1)).type(torch.long)
        pce_mean = 0
        cdf_vals = dist.cdf(y.flatten())
        bin_size = int(cdf_vals.shape[0]/n_bins)
        errs = torch.zeros(n_y_bins, n_bins).to(y.get_device())
        for i in range(n_y_bins):
            offset_idx = torch.randint(low=int(bin_size/2), high=bin_size-1, size=(1,)).item()
            for x in range(n_bins):
                if x == 0:
                    selected_indices = sorted_indices[i, :offset_idx]
                elif x > 0 and x < self.discretization -1:
                    selected_indices = sorted_indices[i, offset_idx + (x-1) * bin_size : offset_idx + (x) * bin_size]
                else:
                    selected_indices = sorted_indices[i,  offset_idx + (x-1)* bin_size:]

                selected_cdf = cdf_vals[selected_indices]
                diff_from_uniform=  torch.abs(torch.sort(selected_cdf)[0] -torch.linspace(0.0, 1.0, selected_cdf.shape[0]).to(y.get_device())).mean()
                errs[i][x] = diff_from_uniform
        return torch.mean(errs)




