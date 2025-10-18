
# AdvDataDrift_BridgeCorrosion

Code and experiments for the paper:

**Data drift mitigation in machine learning for predicting the earthquake performance of corroded bridges.**

> This repository implements drift-aware training and evaluation pipelines (PCA baseline and VAE–GAN / VAE–CGAN / VAE–WGAN variants) to mitigate data drift caused by corrosion progression in bridge structures. It reproduces all analyses and figures used in the paper, including KS/Wasserstein metrics, t-SNE visualization, KDE plots, confidence intervals, and boxplots.

## Citation
If you use this repository, please cite:
```bibtex
@article{chen2025corrosion_drift_mitigation,
  title   = {Data drift mitigation in machine learning for predicting the earthquake performance of corroded bridges},
  author  = {Chen, Mengdie},
  year    = {2025},
  journal = {To appear},
  note    = {Code: https://github.com/chenmengdie3-oss/AdvDataDrift_BridgeCorrosion}
}
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 0) (already included) EX1/EX2-1/EX3-1/EX4 Excel files under data/
# 1) Preprocess (MinMax + optional SMOTE) and export CSVs
python scripts/preprocess_data.py --smote --out_dir data/processed

# 2) Baseline RF: train on EX1 only (1000 trees), test on EX4
python scripts/run_baseline.py --data_dir data/processed --out results/baseline

# 3) Train all generative models on EX1+EX2+EX3, export samples (n=500 each)
bash scripts/run_train_all.sh data/processed/train.csv 8

# 4) Metrics & figures (KS/Wasserstein + t-SNE + KDE + boxplots)
bash scripts/run_all_figures.sh data/processed

# 5) Advanced RF (10k trees): train on (real EX1-3 + synthetic), eval on EX4 & combined
python scripts/run_advanced_model.py --data_dir data/processed --models_dir results --out results/advanced
```

## Data Description
- `EX1.xlsx` → 0-year (no corrosion)
- `EX2-1.xlsx` → 25-year corrosion data
- `EX3-1.xlsx` → 50-year corrosion data
- `EX4.xlsx` → 75-year corrosion data (real testing dataset)

These files represent bridge material degradation over time. The models learn data drift patterns and generate synthetic data for long-term prediction.

## Features
- Drift-aware learning using VAE–GAN, VAE–CGAN, and VAE–WGAN frameworks.
- Evaluation with KS distance, Wasserstein distance, and feature distribution overlap.
- Visualization through t-SNE, KDE, and boxplots.
- Random forest baseline for corrosion-driven drift quantification.
- Configurable pipeline for data preprocessing, training, and result plotting.

## Environment
```bash
pip install -r requirements.txt
# or
conda create -n corrosion-drift python=3.10
conda activate corrosion-drift
pip install -r requirements.txt
```

## License
MIT License © 2025 Mengdie Chen

## Contact
For questions or collaborations, please contact:  
**Mengdie Chen** – [GitHub: chenmengdie3-oss](https://github.com/chenmengdie3-oss)

## Reproducible Run (One Command)
```bash
conda env create -f environment.yml
conda activate corrosion-drift
python main.py --stage full
```

## Figure Map
- t-SNE grid → `scripts/plot_tsne_grid.py`
- KDE & distribution overlap → `scripts/plot_kde_overlap.py`
- CI / Boxplots → `scripts/plot_boxplots.py`
- KS / Wasserstein metrics → `scripts/plot_metrics.py`
- Run all figures → `scripts/run_all_figures.py`

## Benchmarks (Example on EX4)
| Method         | KS ↓ | W-dist ↓ | RF-Acc ↑ |
|----------------|-----:|---------:|---------:|
| Baseline (EX1) | 0.42 | 0.118    | 0.71     |
| VAE-GAN        | 0.27 | 0.073    | 0.81     |
| CGAN           | 0.24 | 0.061    | 0.83     |
| WGAN           | 0.22 | 0.058    | 0.84     |
