# try policy network hidden dim == 512 or tanh: bc_hidden_dim 512 or pi_activation_f 'tanh'
for seed in 0 1 2 3 4
do
        for bc_hidden_dim in 256 512
        do
        for rollout_step in 1000 500
        do
        python main.py --env hopper-medium-replay-v2 \
        --seed $seed \
        --is_state_norm True \
        --rollout_step $rollout_step \
        --bc_hidden_dim $bc_hidden_dim \
        --gpu 0
        done
        done
done