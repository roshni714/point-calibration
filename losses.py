import torch
import torch.distributions as D
import math

class GaussianNLL:
    def __init__(self):
        self.name = "gaussian_nll"

    def __call__(self, y, mu, var):
        sigma = torch.sqrt(var)
        comp = D.Normal(mu, sigma)
        log_prob = comp.log_prob(y)
        nll = -log_prob
        return torch.mean(nll)

class GaussianLaplaceMixtureNLL:
    def __init__(self):
        self.name = "gaussian_laplace_mixture_nll"

    def __call__(self, y, mu, var, loc, scale, weight):
        gaussian_likelihood = (1/torch.sqrt(2 * math.pi * var)) * torch.exp(- 0.5 * torch.pow(y-mu, 2)/var) * weight
        laplace_likelihood = (1/(2 * scale)) * torch.exp(-torch.abs(y - loc)/scale) * (1 - weight)
        nll = -torch.log(torch.clamp(gaussian_likelihood + laplace_likelihood, min=1e-20))
        return torch.mean(nll)

class PointCalibrationLoss:
    def __init__(self, discretization):
        self.name = "pointwise_calibration_loss"
        self.discretization = discretization

    def __call__(self, y, dist):
        n_bins = self.discretization
        n_y_bins = 50
        with torch.no_grad():
            thresholds = torch.FloatTensor(50).uniform_(y.min(), y.max()).to(y.get_device())
            threshold_vals = dist.f.cdf(thresholds.view(-1,1))
            sorted_thresholds, sorted_indices = torch.sort(threshold_vals, dim=1)
        total = 0
        count = 0
        indices= (torch.linspace(0, 1, n_bins, device=y.get_device()) * (y.shape[0]-1)).type(torch.long)
        pce_mean = 0
        cdf_vals = dist.cdf(y.flatten())
        bin_size = int(cdf_vals.shape[0]/n_bins)
        errs = torch.zeros(n_y_bins, n_bins).to(y.get_device())
        for i in range(n_y_bins):
            for x in range(n_bins):
                if x != self.discretization -1:
                    selected_indices = sorted_indices[i, x * bin_size : (x+1) * bin_size]
                else:
                    selected_indices = sorted_indices[i,  x* bin_size:]

                selected_cdf = cdf_vals[selected_indices]
                diff_from_uniform=  torch.abs(torch.sort(selected_cdf)[0] -torch.linspace(0.0, 1.0, selected_cdf.shape[0]).to(y.get_device())).mean()
                errs[i][x] = diff_from_uniform
        return torch.mean(errs)




