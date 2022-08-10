#!/bin/bash 
#SBATCH --job-name=gender101
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 50-00:00:00 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH -a 0-5  ### 0-5
#SBATCH -o gender.eca_nfnet_l2.s101.v%a.out
#SBATCH -e gender.eca_nfnet_l2.s101.v%a.err
module load anaconda3 
conda activate /edgehpc/dept/compbio/users/whu1/envs/tmpenv
python src/main_gender.py --seed 101 --pheno gender --model eca_nfnet_l2 --imshape 390_wx --batch 16 --epoch 17 --lrate 3e-5 --version $SLURM_ARRAY_TASK_ID --subsample 0 --r_degree 33
#################  dm_nfnet_f2  eca_nfnet_l2  regnety_32  xcit_m_384  volo_d2_384  regnet_x_32gf
conda deactivate
