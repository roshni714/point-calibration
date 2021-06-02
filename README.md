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

We also provide the commands for reproducing the MIMIC-III and DHS Asset Wealth experiments. The MIMIC-III
dataset can be obtained upon request [here](https://physionet.org/content/mimiciii/1.4/).
