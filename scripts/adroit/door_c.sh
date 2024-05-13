

for seed in 0 1 2 3
do
    for rollout_step in 50 20 10 5
    do 
    for alpha_bc in 0.1 0.0
    do
    for eval_step in 50 100
    do
        python main.py --env 'door-cloned-v1' \
        --seed $seed \
        --path logs \
        --clip_ratio 0.25 \
        --is_iql True \
        --is_clip_action True \
        --alpha_bc $alpha_bc \
        --pi_activation_f 'relu' \
        --bppo_lr 1e-5 \
        --bppo_steps 10000 \
        --percentage 1.0 \
        --rollout_step $rollout_step \
        --eval_step $eval_step \
        --gpu 0
    done
    done
    done
done
