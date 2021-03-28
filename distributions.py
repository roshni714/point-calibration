import torch


class GaussianLaplaceMixtureDistribution:
    def __init__(self, params):
        self.params = params
        mu, var, loc, scale, self.weight = params
        sigma = torch.sqrt(var)
        self.gaussian_comp = D.Normal(mu, sigma)
        self.laplace_comp = D.Laplace(loc, scale) 
        self.dist_mean = self.mean()
        self.dist_std = self.std()

    def cdf(self, y):
        gaussian_cdf = self.gaussian_comp.cdf(y) * self.weight
        laplace_cdf = self.laplace_comp.cdf(y)  * (1 - self.weight)
        return laplace_cdf + gaussian_cdf

    def mean(self):
        return self.weight  * self.gaussian_comp.mean + (1- self.weight) * self.laplace_comp.mean

    def std(self):
        gaussian_part = self.weight * (torch.pow(self.gaussian_comp.mean, 2) + self.gaussian_comp.variance)
        laplace_part = (1-self.weight) * (torch.pow(self.laplace_comp.mean, 2) + self.laplace_comp.variance)
        total = gaussian_part + laplace_part - torch.pow(self.dist_mean, 2)
        return torch.sqrt(total)

class GaussianDistribution:

    def __init__(self, params):
        self.params = params
        mu, var = params
        sigma = torch.sqrt(var)
        self.gaussian_comp = D.Normal(mu, sigma)
        self.dist_mean = self.mean()
        self.dist_std = self.std()

    def cdf(self, y):
        return self.gaussian_comp.cdf(y)

    def mean(self):
        return self.gaussian_comp.mean

    def std(self):
        return self.gaussian_comp.scale
