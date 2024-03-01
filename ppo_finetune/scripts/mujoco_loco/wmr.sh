

for seed in 0 1 2 3 4
do
    for lr_a in 3e-5
    do
        python main.py \
        --lr_c 2e-4 \
        --lr_a $lr_a \
        --seed $seed \
        --env_name walker2d-medium-replay-v2 \
        --K_epochs 30 \
        --gpu 2
    done
done






