#use tanh
for seed in 0 1 2 3 4
do
    for scale_strategy in 'number' 'dynamic'
    do
        python main.py \
        --lr_c 2e-4 \
        --lr_a 3e-5 \
        --seed $seed \
        --env_name halfcheetah-medium-replay-v2 \
        --mini_batch_size 128 \
        --v_hidden_width 256 \
        --K_epochs 20 \
        --scale_strategy $scale_strategy \
        --gpu 0
    done
done



