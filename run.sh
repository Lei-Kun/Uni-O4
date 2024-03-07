python main_uni-o4.py --env pen-human-v1 \
    --rollout_step 20 \
    --v_steps 500000 \
    --q_bc_steps 500000 \
    --bc_steps 200000 \
    --is_iql True \
    --path logs_first_ft \
    --alpha_bppo 0.1 \
    --data_load_path 'dataset_0808' \
    --omega 0.7 \
    --clip_ratio 0.25 \
    --bppo_steps 10000 \
    --percentage 1.0 \
    --gpu 0
# done