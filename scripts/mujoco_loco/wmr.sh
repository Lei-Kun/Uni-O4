# try v_steps 2000000
for seed in 0 1 2 3 4
do
        for bc_hidden_dim in 256 512
        do
        python main.py --env walker2d-medium-replay-v2 \
        --seed $seed \
        --v_steps 500000 \
        --q_bc_steps 500000 \
        --is_state_norm True \
        --is_iql True \
        --bc_hidden_dim $bc_hidden_dim \
        --gpu 0
        done
done