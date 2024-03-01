
for seed in 0 1 2 3 4
do
        python main.py --env hopper-medium-replay-v2 \
        --seed $seed \
        --is_state_norm True \
        --is_iql True \
        --gpu 0
done