
### full dataset
# python main.py --dataset_nb=5 --exp_no=23 --valid_size=30 --log_frq=100 --batch_size=64 --K=10 --induced_size=500 --retained_size=5000 --ind_frq=4 --nb_steps=400 --gendist_size=3000 --is_train=True --is_gendist=False --is_cluster=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --dataset_nb=5 --exp_no=23 --valid_size=20 --log_frq=40 --batch_size=64 --K=10 --induced_size=500 --ind_frq=4 --nb_steps=600 --gendist_size=3000 --is_train=True --is_gendist=False --is_cluster=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --dataset_nb=5 --exp_no=24	 --valid_size=20 --log_frq=40 --batch_size=64 --K=10 --induced_size=500 --ind_frq=4 --nb_steps=600 --gendist_size=3000 --is_train=True --is_gendist=False --is_cluster=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001

# python main.py --dataset_nb=5 --exp_no=10 --valid_size=20 --log_frq=40 --batch_size=32 --K=10 --induced_size=500 --ind_frq=1000 --nb_steps=600 --gendist_size=3000 --is_train=True --is_gendist=False --is_cluster=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --dataset_nb=5 --exp_no=11 --valid_size=20 --log_frq=40 --batch_size=32 --K=10 --induced_size=500 --ind_frq=1000 --nb_steps=600 --gendist_size=3000 --is_train=True --is_gendist=False --is_cluster=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --dataset_nb=5 --exp_no=12 --valid_size=20 --log_frq=40 --batch_size=32 --K=10 --induced_size=500 --ind_frq=1000 --nb_steps=600 --gendist_size=3000 --is_train=True --is_gendist=False --is_cluster=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001




# generate distribution 
# for value in {0..11}
# do
# echo exp : $value
# python main.py --dataset_nb=0 --class_nb=$value --task_nb=0 --valid_size=30 --log_frq=100 --batch_size=64 --K=10 --induced_size=500 --retained_size=5000 --ind_frq=4 --nb_steps=400 --gendist_size=500 --is_train=False --is_gendist=True --is_cluster=False --is_visualize=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001

# done
# echo finished training 

# ### generate scenarios 
# for value in {1..10}
# do
# echo exp : $value
# python main.py --dataset_nb=1 --scenario_nb=$value --task_nb=0 --valid_size=30 --log_frq=100 --batch_size=64 --K=10 --induced_size=500 --retained_size=5000 --ind_frq=4 --nb_steps=400 --gendist_size=5 --is_train=False --is_gendist=True --is_cluster=False --is_visualize=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001

# done
# echo finished training 


# for value in 1 3 5 11
# do
# echo exp : $value
# python main.py --dataset_nb=$((70+$value)) --exp_no=0 --class_nb=$value --task_nb=0 --valid_size=30 --log_frq=100 --batch_size=64 --K=10 --induced_size=500 --retained_size=5000 --ind_frq=4 --nb_steps=400 --gendist_size=250 --is_train=False --is_gendist=True --is_cluster=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001

# done
# echo finished training 


#### CLUSTER distribution 
# python main.py --dataset_nb=$((10+${SLURM_ARRAY_TASK_ID})) --exp_no=0 --class_nb=${SLURM_ARRAY_TASK_ID} --task_nb=${SLURM_ARRAY_TASK_ID} --valid_size=30 --log_frq=100 --batch_size=64 --K=10 --induced_size=500 --retained_size=5000 --ind_frq=4 --nb_steps=400 --gendist_size=5000 --is_train=False --is_gendist=True --is_cluster=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001 > /dev/null



#### single station training 
# python main.py --exp_type="Gaussian" --gaussian_nb=5 --dataset_nb=100 --exp_no=0 --evolution_nb=1 --class_nb=0 --task_nb=0 --valid_size=50 --log_frq=100 --batch_size=64 --K=10 --induced_size=100 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=1000 --is_train=True --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001


# # station all classes training  
# for value in {0..11}
# do
# echo exp : $value
# python main.py --dataset_nb=0 --class_nb=$value --task_nb=0 --valid_size=5 --log_frq=2 --batch_size=64 --K=2 --induced_size=2 --retained_size=5000 --ind_frq=4 --nb_steps=5 --gendist_size=1000 --is_train=True --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# done
# echo finished training 


# for value in 1 3 5 11
# do
# echo exp : $value
# python main.py --dataset_nb=$((70+$value)) --exp_no=0 --class_nb=$value --task_nb=0 --valid_size=100 --log_frq=2 --batch_size=64 --K=10 --induced_size=50 --retained_size=5000 --ind_frq=4 --nb_steps=3 --gendist_size=1000 --is_train=True --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# done
# echo finished training 


### learn self_drive 

# python main.py --is_selfdrive=True --is_train=False --cont_train=True --valid_size=50 --log_frq=10 --batch_size=64 --induced_size=50 --nb_paramters=3 --nb_steps=600  --learning_rate_t=0.0001 --learning_rate_g=0.0001



####### gaussian all  

# for value in {0..11}
# do
# echo exp : $value
# python main.py --exp_type="Gaussian" --gaussian_nb=$value --dataset_nb=0 --evolution_nb=1 --class_nb=$value --task_nb=0 --valid_size=10 --log_frq=100 --batch_size=64 --K=10 --induced_size=100 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=1000 --is_train=True --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# done
# echo finished training  



# station all classes training  BAYSIAN 
# for value in {0..11}
# do
# echo exp : $value
# python main.py --exp_type="Baysian" --dataset_nb=$((100+$value)) --exp_no=1 --class_nb=$value --task_nb=0 --valid_size=50 --log_frq=100 --batch_size=64 --K=10 --induced_size=100 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=1000 --is_train=True --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# done
# echo finished training 


#########single training  for Evolve 
# python main.py --exp_type="Adversarial" --scenario_nb=10 --dataset_nb=1 --evolution_nb=1 --task_nb=0 --valid_size=20 --log_frq=500	 --batch_size=64 --K=10 --induced_size=500 --retained_size=250 --ind_frq=4 --nb_steps=400 --gendist_size=1000 --is_train=False --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=True --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001



### test all scnearios single station 
# for value in {1..10}
# do
# echo exp : $value
# python main.py --dataset_nb=3 --scenario_nb=$value --is_varsteps=True --valid_size=10 --log_frq=50 --batch_size=64 --K=10 --induced_size=500 --retained_size=5000 --ind_frq=4 --nb_steps=650 --gendist_size=10000 --is_train=True --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0002 --learning_rate_g=0.0002
# done
# echo finished training 

# python main.py --dataset_nb=3 --scenario_nb=1 --is_varsteps=False --valid_size=20 --log_frq=200 --batch_size=64 --K=10 --induced_size=500 --retained_size=5000 --ind_frq=4 --nb_steps=500 --gendist_size=10000 --is_train=True --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=True --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001



#### CLUSTER RANDOMIZED experiment 

# python main.py --exp_type="Adversarial" --scenario_nb=0  --is_randomize=True --dataset_nb=0 --evolution_nb=1 --task_nb=0 --valid_size=250 --log_frq=1000 --batch_size=64 --K=10 --induced_size=900 --retained_size=2500 --ind_frq=4 --nb_steps=400 --gendist_size=1000 --is_train=True --is_gendist=False --is_cluster=True --is_visualize=False --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001


# ### test all classes single station 
# for value in {0..11}
# do
# echo exp : $value
# python main.py --dataset_nb=2 --scenario_nb=0 --is_varsteps=False --class_nb=$value --valid_size=24 --task_nb=0 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=801 --gendist_size=20000 --is_train=True --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# done
# echo finished training 




# for value in 6 7
# do
# echo exp : $value
# python main.py --exp_type="Gaussian" --gaussian_nb=10 --dataset_nb=2 --evolution_nb=1 --class_nb=$value --task_nb=0 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=1000 --is_train=True --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# done
# echo finished training  


# for value in 4 5
# do
# echo exp : $value
# python main.py --exp_type="GP" --gaussian_nb=1 --dataset_nb=3 --evolution_nb=1 --scenario_nb=2 --task_nb=$value --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# done
# echo finished training  


# python main.py --exp_type="GP" --gaussian_nb=1 --dataset_nb=3 --evolution_nb=1 --scenario_nb=1 --task_nb=3 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001

# ### some Baysian
# for value in 3 8
# do
# echo exp : $value
# # python main.py --exp_type="Baysian" --gaussian_nb=5 --dataset_nb=2 --evolution_nb=1 --class_nb=0 --task_nb=0 --valid_size=5 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=False --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# done
# echo finished training 


######## cluster experimtns :

# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=1 --task_nb=3 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=3 --task_nb=1 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=2 --task_nb=4 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=4 --task_nb=2 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=2 --task_nb=5 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=5 --task_nb=2 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=6 --task_nb=8 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=8 --task_nb=6 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=7 --task_nb=9 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=9 --task_nb=7 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=7 --task_nb=10 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001
# python main.py --exp_type="GP" --dataset_nb=4 --scenario_nb=10 --task_nb=7 --valid_size=250 --log_frq=100 --batch_size=64 --K=10 --induced_size=1000 --retained_size=5000 --ind_frq=4 --nb_steps=600 --gendist_size=20000 --is_train=False --is_gendist=False --is_cluster=True --is_visualize=True --cont_train=False --optimize_oracle=False --restore_all=False --is_focal=False --is_evolve=False --keep_bank=False --full_set=True --learning_rate_t=0.0001 --learning_rate_g=0.0001



######## generate set of points 
python main.py --network=resnet --all_points=0 --class_nb=1 --object_nb=1 --iterations=5 #--override=False --reduced=False