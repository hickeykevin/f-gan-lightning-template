#!/bin/bash
#BATCH --job-name=f_gan_dumb_exp
#SBATCH --output=f_gan_output.log
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=128gb
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

RUNPATH=/home/khickey/f-gan-lightning-template/
cd $RUNPATH
source f-gan-env/bin/activate
PYTHONIOENCODING=utf8 python3 run.py experiment=dcfgan trainer.max_epochs=2
