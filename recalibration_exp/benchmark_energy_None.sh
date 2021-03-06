#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --exclude=atlas1,atlas2,atlas3,atlas4,atlas5,atlas6,atlas7,atlas8,atlas9,atlas10,atlas11,atlas12,atlas13,atlas14,atlas15,atlas20,atlas18,atlas17,atlas16
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="benchmark_energy_None.sh"
#SBATCH --error="/atlas/u/rsahoo/point-calibration/recalibration_exp/benchmark_energy_None_err.log"
#SBATCH --output="/atlas/u/rsahoo/point-calibration/recalibration_exp/benchmark_energy_None_out.log"

echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 0 --loss gaussian_laplace_mixture_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 0 --loss gaussian_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 1 --loss gaussian_laplace_mixture_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 1 --loss gaussian_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 2 --loss gaussian_laplace_mixture_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 2 --loss gaussian_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 3 --loss gaussian_laplace_mixture_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 3 --loss gaussian_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 4 --loss gaussian_laplace_mixture_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 4 --loss gaussian_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 5 --loss gaussian_laplace_mixture_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
python /atlas/u/rsahoo/point-calibration/recalibrate.py main --seed 5 --loss gaussian_nll --save baseline  --dataset energy --val_only --cuda --save_dir results
sleep 1
