


for seed in 0 1 2 3 4
do
    for lr_a in 1e-5 3e-5
    do
        for epochs in 20 30 
        python main.py \
        --lr_c 2e-4 \
        --lr_a $lr_a \
        --seed $seed \
        --env_name walker2d-medium-expert-v2 \
        --K_epochs $epochs \
        --gpu 4
    done
done