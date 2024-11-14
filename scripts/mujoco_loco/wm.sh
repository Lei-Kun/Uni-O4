
for seed in 0 1 2 3 4
do
        for bc_hidden_dim in 256 512
        do
        python main.py --env walker2d-medium-v2 \
        --seed $seed \
        --is_state_norm True \
        --is_iql True \
        --bc_hidden_dim $bc_hidden_dim \
        --gpu 0
        done
done