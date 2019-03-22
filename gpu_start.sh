#!/bin/bash -l
#SBATCH --job-name resnet
#SBATCH --array=0-9
#SBATCH --reservation=IVUL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/output.%a.%j.out
#SBATCH -e logs/output.%a.%j.err
#SBATCH --mem 70GB
module purge
module load el7.5/cuda/9.2.148.1
# module load gcc/6.4.0 
# python setup.py install
source activate mytensor

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DIR=/home/hamdiaj/semantic-robustness/
cd $DIR

echo $DIR
echo `pwd`
echo `hostname`

# python main.py --network=resnet --all_points=1 --class_nb=${SLURM_ARRAY_TASK_ID} --object_nb=1 --iterations=600 --override=1 --reduced=0
python main.py --network=resnet --all_points=1 --class_nb=0 --object_nb=1 --iterations=600 --override=1 --reduced=0 --custom_points=1 --custom_list=${SLURM_ARRAY_TASK_ID}

#python main.py --exp_type="GP" --class_nb=${SLURM_ARRAY_TASK_ID} --gaussian_nb=1 --scenario_nb=0  --is_randomize=False --dataset_nb=2 --evolution_nb=1 --task_nb=${SLURM_ARRAY_TASK_ID} --valid_size=250 --log_frq=1000 --batch_size=64 --K=10 --induced_size=1000 --retained_size=2500 --ind_frq=4 --nb_steps=400 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001 > /dev/null
#python main.py --exp_type="Gaussian" --class_nb=${SLURM_ARRAY_TASK_ID} --gaussian_nb=1000 --scenario_nb=0  --is_randomize=False --dataset_nb=2 --evolution_nb=1 --task_nb=${SLURM_ARRAY_TASK_ID} --valid_size=250 --log_frq=1000 --batch_size=64 --K=10 --induced_size=1000 --retained_size=2500 --ind_frq=4 --nb_steps=400 --gendist_size=20000 --is_train=True --is_gendist=False --is_cluster=True --is_visualize=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001 > /dev/null
#python main.py --exp_type="Adversarial" --class_nb=${SLURM_ARRAY_TASK_ID} --scenario_nb=0  --is_randomize=True --dataset_nb=2 --evolution_nb=1 --task_nb=${SLURM_ARRAY_TASK_ID} --valid_size=250 --log_frq=1000 --batch_size=128 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=400 --gendist_size=1000 --is_train=True --is_gendist=False --is_cluster=True --is_visualize=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True  > /dev/null
#python main.py --exp_type="Baysian" --gaussian_nb=500 --dataset_nb=2 --evolution_nb=1 --class_nb=${SLURM_ARRAY_TASK_ID} --task_nb=${SLURM_ARRAY_TASK_ID} --valid_size=250 --log_frq=10 --batch_size=64 --K=10 --induced_size=1000 --retained_size=250 --ind_frq=4 --nb_steps=500 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001 > /dev/null
# python main.py --exp_type="Gaussian" --gaussian_nb=10 --dataset_nb=2 --evolution_nb=1 --class_nb=7 --task_nb=${SLURM_ARRAY_TASK_ID} --valid_size=250 --log_frq=10 --batch_size=64 --K=10 --induced_size=1000 --retained_size=250 --ind_frq=4 --nb_steps=500 --gendist_size=20000 --is_train=True --is_gendist=False --is_cluster=True --is_visualize=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001 > /dev/null
#python main.py --dataset_nb=$((10+${SLURM_ARRAY_TASK_ID})) --exp_no=0 --class_nb=${SLURM_ARRAY_TASK_ID} --task_nb=${SLURM_ARRAY_TASK_ID} --valid_size=30 --bb_log_frq=100 --batch_size=64 --K=10 --induced_size=500 --retained_size=5000 --bb_ind_frq=4 --nb_steps=400 --gendist_size=5000 --is_train=False --is_gendist=True --is_cluster=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001 >/dev/null
#python main.py --dataset_nb=5 --exp_no=65 --task_nb=${SLURM_ARRAY_TASK_ID} --valid_ize=30 --bb_log_frq=100 --batch_size=64 --K=10 --induced_size=500 --retained_size=1000 --bb_ind_frq=4 --nb_steps=600 --gendist_size=3000 --is_train=True --is_gendist=False --is_cluster=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001 >/dev/null
#python main.py --dataset_nb=5 --exp_no=55 --valid_size=20 --bb_log_frq=40 --batch_size=64 --K=10 --induced_size=500 --bb_ind_frq=10000 --nb_steps=600 --gendist_size=3000 --is_train=True --is_gendist=False --is_cluster=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001 
#python main.py --dataset_nb=5 --exp_no=56 --valid_size=20 --bb_log_frq=40 --batch_size=64 --K=10 --induced_size=500 --bb_ind_frq=10000 --nb_steps=600 --gendist_size=3000 --is_train=True --is_gendist=False --is_cluster=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001 

# rm core.*
