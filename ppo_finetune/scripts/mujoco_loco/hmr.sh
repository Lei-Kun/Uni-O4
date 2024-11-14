for seed in 0 1
do
    for lr_a in 1e-5 2e-5
    do 
    for epochs in 5 10
    do
    python main.py \
    --env_name hopper-medium-replay-v2 \
    --lr_a $lr_a \
    --lr_c 2e-4 \
    --v_hidden_width 256 \
    --seed $seed \
    --K_epochs $epochs \
    --scale_strategy 'number' \
    --gpu 0
    done
    done
done 