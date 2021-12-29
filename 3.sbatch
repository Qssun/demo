#!/bin/bash
#SBATCH -n 32
#SBATCH -N 2
#SBATCH --account=yyue
#SBATCH --partition=hpib

echo -n " Start time : " >> time
date >> time

cd /project/qssun/sun/python/demo1224/code
python3.7 train.py

echo -n " End time  : " >> time
date >> time
~