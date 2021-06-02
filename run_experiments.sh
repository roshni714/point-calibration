#!/bin/bash

for experiment in /atlas/u/rsahoo/point-calibration/baseline_model_training/*.sh
do
    echo $experiment
    chmod u+x $experiment
#    sbatch $experiment
    $experiment
    sleep 1
done

for experiment in /atlas/u/rsahoo/point-calibration/recalibration_exp/*.sh
do
    echo $experiment
    chmod u+x $experiment
#    sbatch $experiment
    $experiment
    sleep 1
done

echo "Done"
