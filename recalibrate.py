from pytorch_lightning import Trainer, callbacks
import torch
from modules import PointRecalibrationModel, AverageRecalibrationModel, DistributionRecalibrationModel, GaussianNLLModel, GaussianLaplaceMixtureNLLModel
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from data_loaders import get_credit_regression_dataloader, get_uci_dataloaders, get_satellite_dataloaders
from distributions import GaussianDistribution, GaussianLaplaceMixtureDistribution 
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

def get_baseline_model_predictions(model, dist_class, train, val, test, cuda=False):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    def dataset_dist(data):
        for batch in data:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            params = model(x)
            if cuda:
                params = [param.flatten() for param in params]
                y= y.flatten()
            else:
                params = [param.detach().cpu().flatten() for param in params]
                y = y.flatten().detach().cpu()
            dist = dist_class(tuple(params))
        return dist, y 

    train_dist, y_train = dataset_dist(train)
    val_dist, y_val = dataset_dist(val)
    test_dist, y_test  = dataset_dist(test)
    return train_dist, y_train, val_dist, y_val, test_dist, y_test 

def train_recalibration_model(model, dataset, loss, seed, epochs, actual_datasets = None):

    if "Point" in model.__class__.__name__:
        logger = TensorBoardLogger(
            save_dir="runs", name="logs/{}_{}_seed_{}".format(dataset, loss, seed)
        )
        checkpoint_callback = callbacks.model_checkpoint.ModelCheckpoint(
            "recalibration_models/{}_{}_seed_{}/".format(dataset, loss, seed),
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )
        train, val, test = actual_datasets # hack so can use pytorch lightning for training

        trainer = Trainer(
           gpus=1,
           checkpoint_callback=checkpoint_callback,
           max_epochs=1,
           logger=logger,
           check_val_every_n_epoch=1,
           log_every_n_steps=1
        )
        trainer.fit(model, train_dataloader=train, val_dataloaders=val)
        trainer.test(test_dataloaders=test)
    else:
        model.training_step()
        val_outputs = model.validation_step()
        test_outputs = model.testing_step()
        model.test_epoch_end([test_outputs]) 
    return model




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

    if loss == "gaussian_nll":
        model_class = GaussianNLLModel
        dist_class = GaussianDistribution
        n_in = 3
    elif loss == "gaussian_laplace_mixture_nll":
        model_class = GaussianLaplaceMixtureNLLModel
        dist_class = GaussianLaplaceMixtureDistribution
        n_in = 6
    model_path = "models/{}_{}_seed_{}.ckpt".format(dataset, loss, seed)
    print(model_path),

    if "sigmoid" in posthoc_recalibration:
        recalibration_parameters = {"num_layers": num_layers, "n_dim": n_dim, "epochs": epochs, "n_bins": n_bins, "save_path": "recalibration_models/{}_{}_sigmoid_{}layers_{}dim_{}bins_{}epochs_{}.ckpt".format(dataset, loss, num_layers, n_dim, n_bins, epochs, seed) }
#        if "sigmoid_average" == posthoc_recalibration:
#            recalibration_parameters["n_bins"] = 2
    elif "distribution" in posthoc_recalibration:
        recalibration_parameters = {"n_bins": n_bins}
    else:
        recalibration_parameters = None

    model = model_class.load_from_checkpoint(model_path, input_size=in_size[0], y_scale=y_scale)

    if posthoc_recalibration == "point": 
        dist_datasets = get_baseline_model_predictions(model, dist_class, train, val, test, cuda=True)
        del model
        recalibration_model = PointRecalibrationModel(dist_datasets, n_in=n_in, n_layers=num_layers, n_dim=n_dim, n_bins=n_bins, y_scale=y_scale)
        recalibration_model = train_recalibration_model(recalibration_model, dataset, loss, seed, epochs, actual_datasets=(train, val, test))
    elif posthoc_recalibration == "average":
        dist_datasets = get_baseline_model_predictions(model, dist_class, train, val, test, cuda=False)
        del model
        recalibration_model = AverageRecalibrationModel(dist_datasets, y_scale=y_scale)
        recalibration_model = train_recalibration_model(recalibration_model, dataset, loss, seed, 1)
    elif posthoc_recalibration == "distribution":
        dist_datasets = get_baseline_model_predictions(model, dist_class, train, val, test, cuda=False)
        del model
        recalibration_model = DistributionRecalibrationModel(dist_datasets, y_scale=y_scale, n_bins=n_bins) 
        recalibration_model = train_recalibration_model(recalibration_model, dataset, loss, seed, 1)

    report_recalibration_results(recalibration_model, dataset, train_frac, loss, seed, posthoc_recalibration, recalibration_parameters, save)
 
if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
