#!/bin/bash
datasets=(
"SiChuan"
)
model_types=(
)
for dataset in "${datasets[@]}"; do
        #tune xgboost
        python ./train_model_classical.py --gpu 0 --model_type "xgboost" --dataset "${dataset}" --seed_num 2 --tune  --n_trials 100   > "./log/${dataset}-xgboost-tune.txt"
        #tune catboost
        # python ./train_model_classical.py --gpu 0 --model_type "catboost" --dataset "${dataset}" --seed_num 2 --cat_policy indices --tune --n_trials 100 > "./log/${dataset}-catboost-tune.txt"
        #run with default parameters
        # python ./train_model_classical.py --gpu 0 --model_type "xgboost" --dataset "${dataset}" --seed_num 3   > "./log/${dataset}-xgboost.txt"
        # python ./train_model_classical.py --gpu 0 --model_type "catboost" --dataset "${dataset}" --seed_num 2 --cat_policy indices  > "./log/${dataset}-catboost.txt"
done