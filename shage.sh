#!/bin/bash 
#SBATCH --job-name=age60
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 50-00:00:00 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH -a 0-19  ### 0-5
#SBATCH -o age.regnety_32.s98.sub60.v%a.out
#SBATCH -e age.regnety_32.s98.sub60.v%a.err
module load anaconda3 
conda activate /edgehpc/dept/compbio/users/whu1/envs/tmpenv
python src/main_age.py --seed 98 --model regnety_32 --imshape 390_wx --batch 16 --epoch 17  --lrate 3e-5 --loss_type MAE --pheno age --version $SLURM_ARRAY_TASK_ID --subsample 60 --r_degree 33
#################  dm_nfnet_f2  eca_nfnet_l2  regnety_32  xcit_m_384  volo_d2_384  regnet_x_32gf
conda deactivate
