#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=pubmed_sdne
#SBATCH --mem=128G
#SBATCH --partition=netsi_largemem
#SBATCH --time=80:00:00

conda activate gpu; /home/shafi.z/.conda/envs/gpu/bin/python /home/shafi.z/ExplainingNodeEmbeddings/scripts/sdne_experiments.py --graph_path "/home/shafi.z/ExplainingNodeEmbeddings/data/pubmed.pkl" --run_count 10 --hyp_key "hyp_pubmed" --outfile "../results/pubmed_sdne.pkl" > ../logs/pubmed_sdne.log


 