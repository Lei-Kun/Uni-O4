for seed in 0 1 2 3 4
do
        python main.py --env halfcheetah-medium-replay-v2 \
        --seed $seed \
        --is_iql True \
        --pi_activation_f 'tanh' \
        --gpu 0
done
