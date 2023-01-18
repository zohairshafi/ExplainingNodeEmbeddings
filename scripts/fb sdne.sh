#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=fb_sdne
#SBATCH --mem=128G
#SBATCH --partition=netsi_largemem
#SBATCH --time=80:00:00

conda activate gpu; /home/shafi.z/.conda/envs/gpu/bin/python /home/shafi.z/ExplainingNodeEmbeddings/scripts/sdne_experiment.py --graph_path "/home/shafi.z/ExplainingNodeEmbeddings/data/FB15k.pkl" --run_count 5 --hyp_key "hyp_fb15k"  --outfile "../results/fb_sdne.pkl" > ../logs/fb_sdne.log


 