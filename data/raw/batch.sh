#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=12:00:00
#SBATCH --constraint=centos7
#SBATCH --partition=sched_mit_hill
#SBATCH --mem=4000
#SBATCH -J ions

zsh
source $HOME/.zshrc
module load anaconda
conda activate ion-image

python make_images.py 

