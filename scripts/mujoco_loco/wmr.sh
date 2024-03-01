# try v_steps 2000000
for seed in 0 1 2 3 4
do
        python main.py --env walker2d-medium-replay-v2 \
        --seed $seed \
        --v_steps 500000 \
        --q_bc_steps 500000 \
        --is_state_norm True \
        --is_iql True \
        --gpu 0
done