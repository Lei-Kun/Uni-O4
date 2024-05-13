for seed in 0 1 2 3 
do  
    for step in 5 10 20 
    do
    for lr in 5e-6 3e-5 1e-5 
    do
        python main.py --env 'kitchen-complete-v0' \
                --seed $seed \
                --v_steps 500000 \
                --q_bc_steps 500000 \
                --bc_steps 100000 \
                --rollout_step $step \
                --path logs \
                --bppo_steps 1000 \
                --is_iql True \
                --bc_hidden_dim 256 \
                --bc_depth 3 \
                --bppo_lr $lr \
                --gpu 3
    done
    done
done

