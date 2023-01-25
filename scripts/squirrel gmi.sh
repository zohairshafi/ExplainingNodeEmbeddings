#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --job-name=squirrel_gmi
#SBATCH --mem=64G
#SBATCH --nodelist=c4022
#SBATCH --partition=netsi_largemem
#SBATCH --time=80:00:00

conda activate gpu; /home/shafi.z/.conda/envs/gpu/bin/python /home/shafi.z/ExplainingNodeEmbeddings/scripts/gmi_experiments.py --graph_path "/home/shafi.z/ExplainingNodeEmbeddings/data/squirrel.pkl" --run_count 5 --hyp_key "hyp_squirrel" --use_id 'False' --update_outfile "False" --outfile "../results/squirrel_gmi.pkl" > ../logs/squirrel_gmi.log


 