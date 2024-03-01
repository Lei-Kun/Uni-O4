for seed in 0 1 2 3 4
do
        python main.py --env halfcheetah-medium-v2 \
        --seed $seed \
        --is_state_norm True \
        --is_iql True \
        --pi_activation_f 'tanh' \
        --gpu 0
done