
#!/usr/bin/env bash
set -e
DATA_DIR=${1:-data/processed}
python scripts/plot_tsne_grid.py --real_dir $DATA_DIR --models_dir results --out results/figs
python scripts/plot_metrics.py   --real_dir $DATA_DIR --models_dir results --out results/metrics
python scripts/plot_kde_overlap.py --real_dir $DATA_DIR --models_dir results --out results/figs/kde
python scripts/plot_boxplots.py --metrics_dir results/metrics --out results/figs/boxplots
echo "Figures & metrics generated under results/"
