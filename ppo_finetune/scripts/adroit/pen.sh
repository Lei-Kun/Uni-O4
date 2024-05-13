#for adroit tasks, parameters are needed to be tuned
for seed in 0 1 2 3 4
do
    for epochs in 10 5 8
    do
        for lr_a in 3e-6 2e-6 6e-6
        do
            for strategy in 'dynamic' 'number'
            do
            python main.py \
            --lr_c 2e-4 \
            --lr_a $lr_a \
            --seed $seed \
            --env_name 'pen-human-v1' \
            --mini_batch_size 256 \
            --hidden_width 256 \
            --v_hidden_width 256 \
            --scale_strategy $strategy \
            --K_epochs $epochs \
            --gpu 4
            done
        done
    done
done