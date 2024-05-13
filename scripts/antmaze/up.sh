for seed in 0 1 2
do
        for lr in 5e-5 3e-5 1e-5
        do
        for rollout_step in 50 70 100
        do
        for temperature in 0 10
        do
        for alpha_bc in 0.0 0.1
        do
        python main.py --env antmaze-umaze-diverse-v2 \
        --v_hidden_dim 256 \
        --v_depth 3 \
        --q_hidden_dim 256 \
        --q_depth 3 \
        --decay 0.98 \
        --seed $seed \
        --alpha_bc $alpha_bc \
        --temperature $temperature \
        --path 'logs' \
        --is_iql True \
        --bppo_lr $lr \
        --omega 0.9 \
        --rollout_step $rollout_step \
        --eval_freq 2000 \
        --is_filter_bc True \
        --gpu 2
        done
        done
        done
        done
done
