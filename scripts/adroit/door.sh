for seed in 0 1 2 3
do
    for alpha_bc in 0.1 0.0
    do 
        for rollout_step in 20 30 50 100
        do 
        python main.py --env 'door-human-v1' \
        --seed $seed \
        --path logs \
        --clip_ratio 0.25 \
        --is_iql True \
        --is_clip_action True \
        --bppo_steps 5000 \
        --pi_activation_f 'relu' \
        --bppo_lr 1e-5 \
        --rollout_step $rollout_step \
        --percentage 1.0 \
        --alpha_bc $alpha_bc \
        --eval_step 100 \
        --gpu 1
        done
    done
done

