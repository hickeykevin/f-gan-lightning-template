#!/bin/bash
#BATCH --job-name=f_gan_dumb_exp
#SBATCH --output=f-gan_script_output.log
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=128gb
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -C P100
RUNPATH=/home/khickey/f-gan-lightning-template/
cd $RUNPATH
source f-gan-env/bin/activate
PYTHONIOENCODING=utf8 python3 run.py experiment=walter_gan
