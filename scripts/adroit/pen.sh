


for seed in 0 1 2 3 4
do
    for rollout_step in 50 20 100
    do 
    for alpha_bc in 0.1 0.0
    do
        python main.py --env 'pen-human-v1' \
        --seed $seed \
        --path logs \
        --clip_ratio 0.25 \
        --is_iql True \
        --is_clip_action True \
        --alpha_bc $alpha_bc \
        --pi_activation_f 'relu' \
        --bppo_lr 1e-5 \
        --bppo_steps 10000 \
        --rollout_step $rollout_step \
        --eval_step 100 \
        --gpu 6
    done
    done
done