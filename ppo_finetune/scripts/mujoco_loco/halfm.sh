# use tanh 
for seed in 0 1 2
do
    for lr_a in 3e-5
        do
        python main.py \
        --lr_c 2e-4 \
        --lr_a $lr_a \
        --seed $seed \
        --env_name halfcheetah-medium-v2 \
        --mini_batch_size 128 \
        --v_hidden_width 256 \
        --K_epochs 30 \
        --scale_strategy 'dynamic' \
        --gpu 0
    done
done



