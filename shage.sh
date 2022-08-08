#!/bin/bash 
#SBATCH --job-name=eca_nf
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 50-00:00:00 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH -a 100  ### 0-5
#SBATCH -o age.eca_nfnet_l2.s98.v%a.out
#SBATCH -e age.eca_nfnet_l2.s98.v%a.err
module load anaconda3 
conda activate /edgehpc/dept/compbio/users/whu1/envs/tmpenv
python main_age.py --seed 98 --model eca_nfnet_l2 --imshape 390_wx --batch 16 --epoch 3 --lrate 3e-5 --loss_type MAE --pheno age --version $SLURM_ARRAY_TASK_ID --subsample 0 --r_degree 33
################# python main_gender.py --seed 8 --model dm_nfnet_f3  cait_s24_384  eca_nfnet_l2 epoch=17   --imshape 384 --batch 8 --epoch 50 --lrate 1e-4 --pheno gender --small 0 --version $SLURM_ARRAY_TASK_ID
conda deactivate