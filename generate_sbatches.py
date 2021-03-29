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

OUTPUT_PATH="/atlas/u/rsahoo/point-calibration/slurm"
def generate_baseline_models():
    datasets = ["crime"]
    seeds=[0]
    losses=["gaussian_laplace_mixture_nll", "gaussian_nll"]

    for dataset in datasets:
        for seed in seeds:
            for loss in losses:
                exp_id = 'march28_benchmark_{}_{}_{}'.format(dataset, loss, seed)
                script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                base_cmd = 'python /atlas/u/rsahoo/point-calibration/train_baseline_models.py main '
                with open(script_fn, 'w') as f:
                    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                    new_cmd = base_cmd + "--seed {} --loss {} --save baseline_{} --epochs 100 --dataset {}".format(seed, loss, loss, dataset)
                    print(new_cmd, file=f)
                    print('sleep 1', file=f)
def evaluate(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["gaussian_laplace_mixture_nll"]
    params=[0] 
    discretizations = [0]
#    train_fractions = [0.01, 0.05, 0.1, 0.2, 0.4,1.0]
    n_bins = [20]

#    for disc in discretizations:
#        for param in params:
#            for seed in seeds:
#                for loss in losses:
#                    exp_id = 'feb11_benchmark_{}_{}_{}_{}_{}_{}'.format(dataset, loss, param, None, seed, disc)
#                    script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
#                    base_cmd = 'python /atlas/u/rsahoo/pointwise-calibration/evaluate.py main '
#                    with open(script_fn, 'w') as f:
#                        print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
#                        new_cmd = base_cmd + "--seed {} --loss {} --save {}_eval_old --dataset {} --tradeoff {} --model_size extra_small --discretization {}".format(seed, loss, dataset, dataset, param, disc)
#                        print(new_cmd, file=f)
#                        print('sleep 1', file=f)

    posthoc_recalibration = ["distribution"]
#    binning = [10, 20, 50, 100]
    for seed in seeds:
        for n_bin in n_bins:
            for posthoc_recal in posthoc_recalibration:
                exp_id = 'feb11_benchmark_{}_{}_{}_{}_{}_{}'.format(dataset, "point_calibration_loss", 0.0, posthoc_recal, n_bin, seed)
                script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                base_cmd = 'python /atlas/u/rsahoo/pointwise-calibration/evaluate.py main '
                with open(script_fn, 'w') as f:
                    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                    new_cmd = base_cmd + "--seed {} --loss gaussian_laplace_mixture_nll --save {}_recalibration --dataset {} --tradeoff {} --posthoc_recalibration {} --discretization 0 --n_bins {}".format(seed, dataset,  dataset, 0.0, posthoc_recal, n_bin)
                    print(new_cmd, file=f)
                    print('sleep 1', file=f)


def evaluate_sigmoid(dataset):
    seeds=[0, 1, 2, 3, 4, 5]
    losses=["point_calibration_loss"]
    params=[0] 
    discretizations = [0]

#    for disc in discretizations:
#        for param in params:
#            for seed in seeds:
#                for loss in losses:
#                    exp_id = 'feb11_benchmark_{}_{}_{}_{}_{}_{}'.format(dataset, loss, param, None, seed, disc)
#                    script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
#                    base_cmd = 'python /atlas/u/rsahoo/pointwise-calibration/evaluate.py main '
#                    with open(script_fn, 'w') as f:
#                        print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
#                        new_cmd = base_cmd + "--seed {} --loss {} --save {}_eval_old --dataset {} --tradeoff {} --model_size extra_small --discretization {}".format(seed, loss, dataset, dataset, param, disc)
#                        print(new_cmd, file=f)
#                        print('sleep 1', file=f)

    posthoc_recalibration = ["sigmoid_1D"]
    num_layers = [1]
    n_dims = [100]
    epochs = [20000]
    n_bins = [10, 20, 50, 100]
    train_frac = [0.01, 0.05, 0.1, 0.2, 0.4,1.0]

    for seed in seeds:
        for posthoc_recal in posthoc_recalibration:
            for n_layer in num_layers:
                for n_bin in n_bins:
                    for n_dim in n_dims:
                        exp_id = 'feb11_benchmark_{}_{}_{}_{}_{}_{}_{}_{}'.format(dataset, "point_calibration_loss", 0.0, posthoc_recal, seed, n_layer, n_dim, n_bin)
                        script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
                        base_cmd = 'python /atlas/u/rsahoo/pointwise-calibration/evaluate.py main '
                        with open(script_fn, 'w') as f:
                            print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
                            new_cmd = base_cmd + "--seed {} --loss point_calibration_loss --save {}_sigmoid_separate_mlp --dataset {} --tradeoff {} --model_size extra_small --posthoc_recalibration {} --discretization 0 --epochs 20000 --num_layers {} --n_dim {} --n_bins {}".format(seed, dataset,  dataset, 0.0, posthoc_recal, n_layer, n_dim, n_bin)
                            print(new_cmd, file=f)
                            print('sleep 1', file=f)


generate_baseline_models()
