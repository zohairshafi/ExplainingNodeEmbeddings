#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --job-name=squirrel_dgi
#SBATCH --mem=64G
#SBATCH -w=c4022
#SBATCH --partition=netsi_standard
#SBATCH --time=80:00:00

conda activate gpu; /home/shafi.z/.conda/envs/gpu/bin/python /home/shafi.z/ExplainingNodeEmbeddings/scripts/dgi_experiments.py --graph_path "/home/shafi.z/ExplainingNodeEmbeddings/data/squirrel.pkl" --run_count 5 --hyp_key "hyp_squirrel" --use_id 'False' --update_outfile "False" --outfile "../results/squirrel_dgi.pkl" > ../logs/squirrel_dgi.log


 