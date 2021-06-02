# Reliable Decisions with Threshold Calibration

This is a repository for reproducing the experiments from Reliable Decisions with Threshold Calibration.

## Reproducibility

We provide the commands to reproduce the UCI regression experiments from the paper.

```bash
conda env create -f point-calibration/environment.yml
conda activate calibration
python download_datasets.py
chmod +x run_experiments.sh
./run_experiments
```
