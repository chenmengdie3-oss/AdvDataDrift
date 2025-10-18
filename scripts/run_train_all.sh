
#!/usr/bin/env bash
set -e
DATA=${1:-data/processed/train.csv}
EPOCHS=${2:-8}
python -m src.train_vae --data $DATA --epochs $EPOCHS --out results/vae
python -m src.train_gan --data $DATA --gan_type gan --epochs $EPOCHS --out results/gan
python -m src.train_gan --data $DATA --gan_type cgan --epochs $EPOCHS --out results/cgan
python -m src.train_gan --data $DATA --gan_type wgan-gp --epochs $EPOCHS --out results/wgan_gp
python -m src.train_hybrid --data $DATA --hybrid vae-gan --epochs $EPOCHS --out results/vae_gan
python -m src.train_hybrid --data $DATA --hybrid vae-cgan --epochs $EPOCHS --out results/vae_cgan
echo "All models trained -> results/"
