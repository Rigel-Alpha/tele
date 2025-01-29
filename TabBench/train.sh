#!/bin/bash
datasets=(
"SiChuan_avg"
)
model_types=(
)
for dataset in "${datasets[@]}"; do
        #tune
        python ./train_model_classical.py --gpu 0 --model_type "xgboost" --dataset "${dataset}" --seed_num 2 --tune  --n_trials 100 --retune  > "./log/${dataset}-xgboost-tune.txt"
#        python ./train_model_classical.py --gpu 0 --model_type "catboost" --dataset "${dataset}" --seed_num 2 --cat_policy indices --tune --n_trials 100 --retune > "./log/${dataset}-catboost-tune.txt"
#        python ./train_model_classical.py --gpu 0 --model_type "lightgbm" --dataset "${dataset}" --seed_num 2 --tune  --n_trials 100 --retune  > "./log/${dataset}-lightgbm-tune.txt"
#        python ./train_model_deep.py --gpu 0 --model_type "mlp" --dataset "${dataset}" --seed_num 1 --tune  --n_trials 1 --retune > "./log/${dataset}-mlp-tune.txt"
#        python ./train_model_deep.py --gpu 0 --model_type "resnet" --dataset "${dataset}" --seed_num 2 --tune  --n_trials 100 --retune > "./log/${dataset}-resnet-tune.txt"
#        python ./train_model_deep.py --gpu 0 --model_type "tabr" --dataset "${dataset}" --seed_num 2 --tune  --n_trials 100 --retune  > "./log/${dataset}-tabr-tune.txt"

        #run with default parameters
        # python ./train_model_classical.py --gpu 0 --model_type "xgboost" --dataset "${dataset}" --seed_num 3   > "./log/${dataset}-xgboost.txt"
        # python ./train_model_classical.py --gpu 0 --model_type "catboost" --dataset "${dataset}" --seed_num 2 --cat_policy indices  > "./log/${dataset}-catboost.txt"
done