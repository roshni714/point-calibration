import torch
RANGE = [-10, 10]
RESOLUTION=8000
from scipy.integrate import simps



class CompositionDist():
     def __init__(self, r, f):
        self.r = r
        self.f = f

     def cdf(self, y):
        inner = self.f.cdf(y)
        info = torch.cat(tuple([param.unsqueeze(dim=0) for param in self.f.params]), dim=0)
#        info = torch.cat((self.f.mean.unsqueeze(dim=0), self.f.scale.unsqueeze(dim=0)), dim=0)
        out = self.r.cdf(inner, info)
        assert inner.shape[-1] == info.shape[-1]
        return out

     def dist_mean(self):
         y = torch.linspace(-10, 10, 8000).to(r.get_device())
         out = self.cdf(y).detach().cpu().numpy()
         first_mom = []
         idx = int(self.xs.shape[0]/2)
         for cdf in out:
             l = -simps(cdf[:idx], self.xs[:idx])
             r = simps(1-cdf[idx:], self.xs[idx:]) 
             first_mom.append(l + r)
         return torch.tensor(first_mom)

     def to(self, device):
         self.r = self.r.to(device)
         self.f = self.f.to(device)
         return self

     def detach(self):
         self.r = self.r.cpu()
         self.f = self.f.detach()
         return self




