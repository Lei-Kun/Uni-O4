for seed in 0 1 2 3 
do  
        for step in 2 4 6 8
        do
        for lr in 3e-5 1e-5
        do
        python main.py --env 'kitchen-partial-v0' \
                --seed $seed \
                --v_steps 1000000 \
                --q_bc_steps 1000000 \
                --bc_steps 200000 \
                --is_state_norm True \
                --rollout_step $step \
                --path logs \
                --is_iql True \
                --bppo_steps 1000 \
                --bc_hidden_dim 256 \
                --bc_depth 3 \
                --omega 0.7 \
                --bppo_lr $lr \
                --gpu 2
        done
        done
done