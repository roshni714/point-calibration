from pytorch_lightning import Trainer, callbacks
import csv
import torch
import os
from modules import PointwiseCalibrationModel
from losses import ComboLoss
import numpy as np
from data_loaders import get_credit_regression_dataloader, get_uci_dataloaders, get_satellite_dataloaders
import torch.distributions as distributions
import argh
from reporting import report_recalibration_results

def get_dataset(dataset, seed, train_frac):
    batch_size = None
    if dataset == "satellite":
        train, val, test, in_size, output_size, y_scale = get_satellite_dataloaders(split_seed=seed, batch_size=batch_size)
    else:
        train, val, test, in_size, output_size, y_scale = get_uci_dataloaders(
            dataset, split_seed=seed, test_fraction=0.3, batch_size=batch_size, train_frac=train_frac)
    return train, val, test, in_size,  y_scale

def get_baseline_model_predictions(model, dist_class, train, val, test):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    def dataset_dist(data):
        for batch in data:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            params = model(x)
            params = [param.flatten() for param in params] 
            dist = dist_class(tuple(train_params))
        return dist, y 

    train_dist, y_train = dataset_dist(train)
    val_dist, y_val = dataset_dist(val)
    test_dist, y_test  = dataset_dist(test)
    return train_dist, y_train, val_dist, y_val, test_dist, y_test 

@argh.arg("--seed", default=0)
@argh.arg("--dataset", default="protein")
@argh.arg("--loss", default="gaussian_nll")
@argh.arg("--save", default="protein_evaluation")
@argh.arg("--posthoc_recalibration", default=None)
@argh.arg("--train_frac", default=1.0)

## Recalibration parameters
@argh.arg("--num_layers", default=2)
@argh.arg("--n_dim", default=100)
@argh.arg("--epochs", default=500)
@argh.arg("--n_bins", default=20)


def main(dataset="protein", seed=0, save="real", loss="point_calibration_loss", posthoc_recalibration=None, train_frac=1.0, num_layers=2, n_dim=100, epochs=500, n_bins=20):
    train, val, test, in_size, y_scale = get_dataset(dataset, seed, train_frac)

    if loss_name == "gaussian_nll":
        model_class = GaussianNLLModel
        dist_class = GaussianDistribution
    else:
        model_class = GaussianLaplaceMixtureNLLModel
        dist_class = GaussianLaplaceMixtureDistribution

    model_path = "models/{}_{}_seed_{}.ckpt".format(dataset, loss, seed),

    if "sigmoid" in posthoc_recalibration:
        recalibration_parameters = {"num_layers": num_layers, "n_dim": n_dim, "epochs": epochs, "n_bins": n_bins, "save_path": "recalibration_models/{}_{}_sigmoid_{}layers_{}dim_{}bins_{}epochs_{}.ckpt".format(dataset, loss, num_layers, n_dim, n_bins, epochs, seed) }
#        if "sigmoid_average" == posthoc_recalibration:
#            recalibration_parameters["n_bins"] = 2
    elif "distribution" in posthoc_recalibration:
        recalibration_parameters = {"n_bins": n_bins}
    else:
        recalibration_parameters = None

    model = model_class.load_from_checkpoint(model_path, input_size=in_size[0], y_scale=y_scale)
    dist_datasets = get_baseline_model_predictions(model, dist_class, train, val, test)
    del model

    if posthoc_recalibration == "point": 
        recalibration_model = PointRecalibrationModel(dist_datasets, n_in=n_in, num_layers=num_layers, n_dim=n_dim, n_bins=n_bins)
    elif posthoc_recalibration == "average":
        recalibration_model = AverageRecalibrationModel(dist_datasets)
    elif posthoc_recalibration == "distribution":
        recalibration_model = DistributionRecalibrationModel(dist_datasets) 

    report_recalibration_results(recalibration_model, dataset, train_frac, loss, seed, posthoc_recalibration, save)
 
if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
