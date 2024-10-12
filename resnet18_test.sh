#!/bin/bash
#SBATCH --job-name=mil-resnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

source /media02/tdhoang01/.virtualenvs/myenv/bin/activate
python3 /media02/tdhoang01/python-debugging/rsna/resnet18-test.py