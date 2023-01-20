#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --job-name=citeseer_gmi
#SBATCH --mem=32G
#SBATCH --partition=netsi_standard
#SBATCH --time=80:00:00

conda activate gpu; /home/shafi.z/.conda/envs/gpu/bin/python /home/shafi.z/ExplainingNodeEmbeddings/scripts/gmi_experiments.py --graph_path "/home/shafi.z/ExplainingNodeEmbeddings/data/citeseer.pkl" --run_count 5 --hyp_key "hyp_citeseer" --use_id 'False' --update_outfile "False" --outfile "../results/citeseer_gmi.pkl" > ../logs/citeseer_gmi.log


 