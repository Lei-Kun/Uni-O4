
for seed in 0 1 2 3 4
do
        for bc_hidden_dim in 256 512
        do
        python main.py --env hopper-medium-replay-v2 \
        --seed $seed \
        --is_state_norm True \
        --bc_hidden_dim $bc_hidden_dim \
        --is_iql True \
        --gpu 0
        done
done