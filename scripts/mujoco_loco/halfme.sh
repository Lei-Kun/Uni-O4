for seed in 0 1 2 3 4
do
        python main.py --env halfcheetah-medium-expert-v2 \
        --seed $seed \
        --is_state_norm True \
        --gpu 0
done
