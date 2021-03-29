import csv
import os

def write_result(results_file, result):
    """Writes results to a csv file."""
    with open(results_file, "a+", newline="") as csvfile:
        field_names = result.keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.stat(results_file).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)

def report_baseline_results(model, dataset, train_frac, loss_name, seed, save):
    result = {"dataset":dataset,
              "rmse": model.rmse,
              "loss": loss_name,
              "ece": getattr(model, "ece", 0),
#              "stddev": getattr(model, "sharpness", 0),
#              "point_unbiasedness_max": getattr(model, "point_unbiasedness_max", 0),
#              "point_unbiasedness_mean": getattr(model, "point_unbiasedness_mean", 0),
              "point_calibration_error": getattr(model, "point_calibration_error", 0),
#              "false_positive_rate_error": getattr(model, "false_positive_rate_error", 0),
#              "false_negative_rate_error": getattr(model, "false_negative_rate_error", 0),
              "true_vs_pred_loss": getattr(model, "true_vs_pred_loss", 0),
              "decision_loss": getattr(model, "decision_loss", 0),
#              "posthoc_recalibration": posthoc_recalibration,
              "test_nll": getattr(model, "test_nll", 0),
              "train_frac": train_frac,
#              "learning_rate": learning_rate,
              "seed": seed}
    results_file = "results/" + save + ".csv"
    write_result(results_file, result)
    
    all_err = getattr(model, "all_err", [])
    all_loss = getattr(model, "all_loss", [])
    all_y0 = getattr(model, "all_y0", [])
    all_c = getattr(model, "all_c", [])

    decision_making_results_file = "results/" + save + "_decision_{}_{}_{}.csv".format(dataset, loss_name, seed)
    for i in range(len(all_err)):
        decision_making_dic = {}
        decision_making_dic["y0"] = all_y0[i].item()
        decision_making_dic["decision_loss"] = all_loss[i].item()
        decision_making_dic["err"] = all_err[i].item()
        decision_making_dic["c"] = all_c[i].item()
        write_result(decision_making_results_file, decision_making_dic)

def report_recalibration_results(model, dataset, train_frac, loss_name, seed, posthoc_recalibration, save):
    
    method_name = loss
    results_file = "final_results/" + save + ".csv"
    result = {"dataset":dataset,
              "rmse": model.rmse,
              "loss": loss_name,
              "ece": getattr(model, "ece", 0),
#              "stddev": getattr(model, "sharpness", 0),
#              "point_unbiasedness_max": getattr(model, "point_unbiasedness_max", 0),
#              "point_unbiasedness_mean": getattr(model, "point_unbiasedness_mean", 0),
              "point_calibration_error": getattr(model, "point_calibration_error", 0),
              "point_calibration_error_uniform_mass": getattr(model, "point_calibration_error_uniform_mass", 0),
#              "false_positive_rate_error": getattr(model, "false_positive_rate_error", 0),
#              "false_negative_rate_error": getattr(model, "false_negative_rate_error", 0),
              "true_vs_pred_loss": getattr(model, "true_vs_pred_loss", 0),
              "decision_loss": getattr(model, "decision_loss", 0),
              "posthoc_recalibration": posthoc_recalibration,
              "train_frac": train_frac,
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
        decision_making_results_file = "results/" + save + "_{}_{}_{}_decision.csv".format(loss_name, posthoc_recalibration, seed)
    else:
        decision_making_results_file = "results/" + save + "_{}_{}_decision.csv".format(loss_name, seed)

    for i in range(len(all_err)):
        decision_making_dic = {}
        decision_making_dic["y0"] = all_y0[i].item()
        decision_making_dic["decision_loss"] = all_loss[i].item()
        decision_making_dic["err"] = all_err[i].item()
        decision_making_dic["c"] = all_c[i].item()
        write_result(decision_making_results_file, decision_making_dic)
