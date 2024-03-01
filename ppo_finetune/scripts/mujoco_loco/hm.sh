for seed in 0 1 2 3 4
do
    for lr_a in 2e-5
    do 
    python main.py \
    --env_name hopper-medium-v2 \
    --lr_a $lr_a \
    --lr_c 2e-4 \
    --hidden_width $hidden_width \
    --v_hidden_width 256 \
    --seed $seed \
    --K_epochs 10 \
    --scale_strategy 'number' \
    --gpu 0
    done
done 