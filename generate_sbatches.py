import itertools
import glob
import os


SBATCH_PREFACE = \
"""#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --exclude=atlas1,atlas2,atlas3,atlas4,atlas5,atlas6,atlas7,atlas8,atlas9,atlas10,atlas11,atlas12,atlas13,atlas14,atlas15,atlas20,atlas18,atlas17,atlas16
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
"""

# constants for commands

OUTPUT_PATH="/atlas/u/rsahoo/point-calibration/slurm_distribution"

def generate_baseline_models():
    datasets = ["crime", "kin8nm", "naval", "protein", "satellite"]
#    datasets = ["crime"]
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_laplace_mixture_nll", "gaussian_nll"]

    for dataset in datasets:
        for seed in seeds:
            for loss in losses:
                exp_id = 'march28_benchmark_{}_{}_{}'.format(dataset, loss, seed)
                script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                base_cmd = 'python /atlas/u/rsahoo/point-calibration/train_baseline_models.py main '
                with open(script_fn, 'w') as f:
                    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                    new_cmd = base_cmd + "--seed {} --loss {} --save baseline --epochs 100 --dataset {}".format(seed, loss,  dataset)
                    print(new_cmd, file=f)
                    print('sleep 1', file=f)

def evaluate_average_calibration(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_laplace_mixture_nll", "gaussian_nll"]

    for seed in seeds:
        for loss in losses:
            exp_id = 'mar28_benchmark_{}_{}_{}_{}'.format(dataset, loss, "average", seed)
            script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
            base_cmd = 'python /atlas/u/rsahoo/point-calibration/recalibrate.py main '
            with open(script_fn, 'w') as f:
                print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                new_cmd = base_cmd + "--seed {} --loss {} --save recalibration --dataset {} --posthoc_recalibration average".format(seed, loss, dataset)
                print(new_cmd, file=f)
                print('sleep 1', file=f)

def evaluate_average_calibration_train_frac(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_laplace_mixture_nll", "gaussian_nll"]
    train_fracs = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]

    for seed in seeds:
        for loss in losses:
            for train_frac in train_fracs:
                exp_id = 'mar30_benchmark_{}_{}_{}_{}_{}'.format(dataset, loss, "average", train_frac, seed)
                script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                base_cmd = 'python /atlas/u/rsahoo/point-calibration/recalibrate.py main '
                with open(script_fn, 'w') as f:
                    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                    new_cmd = base_cmd + "--seed {} --loss {} --save recalibration --dataset {} --posthoc_recalibration average --train_frac {}".format(seed, loss, dataset, train_frac)
                    print(new_cmd, file=f)
                    print('sleep 1', file=f)



def evaluate_distribution_calibration(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_laplace_mixture_nll", "gaussian_nll"]
    n_bins = [5]

    for n_bin in n_bins:
        for loss in losses:
            exp_id = 'apr1_benchmark_{}_{}_{}_{}'.format(dataset, loss, "distribution", n_bin)
            script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
            base_cmd = 'python /atlas/u/rsahoo/point-calibration/recalibrate.py main '
            with open(script_fn, 'w') as f:
               print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
               for seed in seeds:
                   new_cmd = base_cmd + "--seed {} --loss {} --save distribution_recalibration --dataset {} --posthoc_recalibration distribution --n_bins {}".format(seed, loss, dataset, n_bin)
                   print(new_cmd, file=f)
               print('sleep 1', file=f)

def evaluate_distribution_calibration_train_frac(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_laplace_mixture_nll", "gaussian_nll"]
    train_fracs = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
    
    for seed in seeds:
        exp_id = 'apr1_benchmark_{}_{}_{}_{}_{}'.format(dataset, loss, "distribution", n_bin, seed)
        script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
        base_cmd = 'python /atlas/u/rsahoo/point-calibration/recalibrate.py main '
        with open(script_fn, 'w') as f:
            for train_frac in train_fracs:
                for loss in losses:
                    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                    new_cmd = base_cmd + "--seed {} --loss {} --save distribution_recalibration --dataset {} --posthoc_recalibration distribution --n_bins {} --train_frac {}".format(seed, loss, dataset, 10, train_frac)
                    print(new_cmd, file=f)
            print('sleep 1', file=f)



def evaluate_point_calibration(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_laplace_mixture_nll", "gaussian_nll"]
    num_layers = [1]
    n_dims = [20]
    epochs = [5000]
    n_bins = [50, 20, 10]

    for n_bin in n_bins:
        for loss in losses:
            for n_layer in num_layers:
                exp_id = 'apr1_benchmark_{}_{}_{}_{}'.format(dataset, loss, n_layer, n_bin)
                script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                base_cmd = 'python /atlas/u/rsahoo/point-calibration/recalibrate.py main '
                with open(script_fn, 'w') as f:
                    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                    for seed in seeds:
                        for n_dim in n_dims:
                            new_cmd = base_cmd + "--seed {} --loss {} --save point_recalibration --dataset {} --posthoc_recalibration point --epochs 5000 --num_layers {} --n_dim {} --n_bins {}".format(seed, loss, dataset, n_layer, n_dim, n_bin)
                            print(new_cmd, file=f)
                    print('sleep 1', file=f)


def evaluate_point_calibration_single_mlp(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_nll", "gaussian_laplace_mixture_nll"]
    num_layers = [1]
    n_dims = [20]
    epochs = [5000]
    n_bins = [10, 20, 50]

    for n_bin in n_bins:
        for loss in losses:
            for n_layer in num_layers:
                exp_id = 'apr1_benchmark_{}_{}_{}_{}'.format(dataset, loss, n_layer, n_bin)
                script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                base_cmd = 'python /atlas/u/rsahoo/point-calibration/recalibrate.py main '
                with open(script_fn, 'w') as f:
                    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                    for seed in seeds:
                        for n_dim in n_dims:
                            new_cmd = base_cmd + "--seed {} --loss {} --save point_single_mlp --dataset {} --posthoc_recalibration point --epochs 5000 --num_layers {} --n_dim {} --n_bins {} --flow_type single_mlp".format(seed, loss, dataset, n_layer, n_dim, n_bin)
                            print(new_cmd, file=f)
                    print('sleep 1', file=f)

def evaluate_point_calibration_single_mlp_learning_rate(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_nll", "gaussian_laplace_mixture_nll"]
    num_layers = [1]
    n_dims = [20]
    epochs = [5000]
    n_bins = [20]

    for n_bin in n_bins:
        for loss in losses:
            for n_layer in num_layers:
                exp_id = 'apr1_benchmark_{}_{}_{}_{}'.format(dataset, loss, n_layer, n_bin)
                script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                base_cmd = 'python /atlas/u/rsahoo/point-calibration/recalibrate.py main '
                with open(script_fn, 'w') as f:
                    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                    for seed in seeds:
                        for n_dim in n_dims:
                            new_cmd = base_cmd + "--seed {} --loss {} --save point_single_mlp_new_lr --dataset {} --posthoc_recalibration point --epochs 5000 --num_layers {} --n_dim {} --n_bins {} --flow_type single_mlp --learning_rate 1e-2".format(seed, loss, dataset, n_layer, n_dim, n_bin)
                            print(new_cmd, file=f)
                    print('sleep 1', file=f)


def evaluate_point_calibration_single_mlp_dropout():
    datasets = ["kin8nm", "satellite", "crime"]
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_nll", "gaussian_laplace_mixture_nll"]
    num_layers = [1]
    n_dims = [20]
    epochs = [1000]
    n_bins = [5, 10, 20]

    for dataset in datasets:
        for n_bin in n_bins:
            for loss in losses:
                for n_layer in num_layers:
                    exp_id = 'apr1_benchmark_{}_{}_{}_{}'.format(dataset, loss, n_layer, n_bin)
                    script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                    base_cmd = 'python /atlas/u/rsahoo/point-calibration/recalibrate.py main '
                    with open(script_fn, 'w') as f:
                        print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                        for seed in seeds:
                            for n_dim in n_dims:
                                 new_cmd = base_cmd + "--seed {} --loss {} --save point_single_mlp_dropout --dataset {} --posthoc_recalibration point --epochs 2000 --num_layers {} --n_dim {} --n_bins {} --flow_type single_mlp_dropout".format(seed, loss, dataset, n_layer, n_dim, n_bin)
                                 print(new_cmd, file=f)
                        print('sleep 1', file=f)


def evaluate_point_calibration_single_mlp_combine(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_nll", "gaussian_laplace_mixture_nll"]
    num_layers = [1]
    n_dims = [20]
    epochs = [5000]
    n_bins = [20]

    for n_bin in n_bins:
        for loss in losses:
            for n_layer in num_layers:
                exp_id = 'apr1_benchmark_{}_{}_{}_{}'.format(dataset, loss, n_layer, n_bin)
                script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                base_cmd = 'python /atlas/u/rsahoo/point-calibration/recalibrate.py main '
                with open(script_fn, 'w') as f:
                    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                    for seed in seeds:
                        for n_dim in n_dims:
                            new_cmd = base_cmd + "--seed {} --loss {} --save point_single_mlp --dataset {} --posthoc_recalibration point --epochs 5000 --num_layers {} --n_dim {} --n_bins {} --flow_type single_mlp --combine_val_train --learning_rate 1e-2".format(seed, loss, dataset, n_layer, n_dim, n_bin)
                            print(new_cmd, file=f)
                    print('sleep 1', file=f)



#generate_baseline_models()
#evaluate_distribution_calibration("crime")
#for dataset in ["naval", "crime", "satellite", "kin8nm", "protein"]:
#    evaluate_distribution_calibration(dataset)

for dataset in ["naval", "protein", "crime", "kin8nm", "satellite"]:
    evaluate_distribution_calibration(dataset)
