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

def write_result(results_file, result):
    """Writes results to a csv file."""
    with open(results_file, "a+", newline="") as csvfile:
        field_names = result.keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.stat(results_file).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)

@argh.arg("--seed", default=0)
@argh.arg("--dataset", default="protein")
@argh.arg("--loss", default="gaussian_nll")
@argh.arg("--save", default="protein_evaluation")
@argh.arg("--posthoc_recalibration", default=None)
@argh.arg("--train_frac", default=1.0)
@argh.arg("--discretization", default=200)

## Recalibration parameters
@argh.arg("--num_layers", default=2)
@argh.arg("--n_dim", default=100)
@argh.arg("--epochs", default=500)
@argh.arg("--n_bins", default=20)

def main(dataset="protein", seed=0, save="real", loss="point_calibration_loss", posthoc_recalibration=None, train_frac=1.0, num_layers=2, n_dim=100, epochs=500, n_bins=20):
    batch_size = None
    if dataset == "credit":
        train, val, test, in_size, output_size, y_scale = get_credit_regression_dataloader(split_seed=seed, batch_size=batch_size)
    elif dataset== "satellite":
        train, val, test, in_size, output_size, y_scale = get_satellite_dataloaders(split_seed=seed, batch_size=batch_size)
    else:
        train, val, test, in_size, output_size, y_scale = get_uci_dataloaders(
            dataset, split_seed=seed, test_fraction=0.3, batch_size=None, train_frac=train_frac)

    loss_name = loss
    if loss_name == "gaussian_nll":
        model_class = GaussianNLLModel
    else:
        model_class = GaussianLaplaceMixtureNLLModel
    model_path = "models/{}_{}_{}_{}_{}_{}_seed_{}.ckpt".format(dataset, loss, float(tradeoff), seed, "None", discretization, seed)

    loss = ComboLoss(loss_name, tradeoff=tradeoff, discretization=50)

    if "sigmoid" in posthoc_recalibration:
        recalibration_parameters = {"num_layers": num_layers, "n_dim": n_dim, "epochs": epochs, "n_bins": n_bins, "save_path": "recalibration_models/{}_sigmoid_{}layers_{}dim_{}bins_{}epochs_{}.ckpt".format(dataset, num_layers, n_dim, n_bins, epochs, seed) }
        if "sigmoid_average" == posthoc_recalibration:
            recalibration_parameters["n_bins"] = 2
    elif "distribution" in posthoc_recalibration:
        recalibration_parameters = {"n_bins": n_bins}
    else:
        recalibration_parameters = None
    model = PointwiseCalibrationModel.load_from_checkpoint(model_path, input_size=in_size[0], y_scale=y_scale, loss=loss, model_size=model_size, posthoc_recalibration=posthoc_recalibration, validation_dataloader=train, recalibration_parameters=recalibration_parameters)
    trainer = Trainer(
        gpus=1,
    )
#    trainer.test(model, test_dataloaders=test)
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    for i, batch in enumerate(test):
        model.recalibrate(batch, i)

    method_name = loss
    results_file = "final_results/" + save + ".csv"
    result = {"dataset":dataset,
              "rmse": model.rmse,
              "loss": loss_name,
              "discretization": discretization,
              "tradeoff": tradeoff,
              "ece": getattr(model, "ece", 0),
              "stddev": getattr(model, "sharpness", 0),
              "point_unbiasedness_max": getattr(model, "point_unbiasedness_max", 0),
              "point_unbiasedness_mean": getattr(model, "point_unbiasedness_mean", 0),
              "point_calibration_error": getattr(model, "point_calibration_error", 0),
              "point_calibration_error_uniform_mass": getattr(model, "point_calibration_error_uniform_mass", 0),
              "false_positive_rate_error": getattr(model, "false_positive_rate_error", 0),
              "false_negative_rate_error": getattr(model, "false_negative_rate_error", 0),
              "true_vs_pred_loss": getattr(model, "true_vs_pred_loss", 0),
              "decision_loss": getattr(model, "decision_loss", 0),
              "decision_gap": getattr(model, "decision_gap", 0),
              "posthoc_recalibration": posthoc_recalibration,
              "train_frac": train_frac,
              "learning_rate": 1e-3,
              "seed": seed}
    for x in ["num_layers", "n_dim", "epochs", "n_bins"]:
        if recalibration_parameters and x in recalibration_parameters:
            result[x] = recalibration_parameters[x]
        else:
            result[x] = None
    write_result(results_file, result)

    all_err = getattr(model, "all_err", [])
    all_loss = getattr(model, "all_loss", [])
    all_y0 = getattr(model, "all_y0", [])
    all_c = getattr(model, "all_c", [])

    if posthoc_recalibration:
        decision_making_results_file = "final_results/" + save + "_{}_{}_{}_{}_decision.csv".format(loss_name, tradeoff, posthoc_recalibration, seed)
    else:
        decision_making_results_file = "final_results/" + save + "_{}_{}_{}_decision.csv".format(loss_name, tradeoff, seed)

    for i in range(len(all_err)):
        decision_making_dic = {}
        decision_making_dic["y0"] = all_y0[i].item()
        decision_making_dic["decision_loss"] = all_loss[i].item()
        decision_making_dic["err"] = all_err[i].item()
        decision_making_dic["c"] = all_c[i].item()
        write_result(decision_making_results_file, decision_making_dic)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
