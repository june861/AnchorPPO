update_epochs=(5 10 15 20)
algos=(
    "ppo"
    "appo"
    "appo-pow"
    "spo"
)
envs=(
    "AssaultNoFrameskip-v4"
)
seeds=(1 2 3 4 5)
project="atari-v1"
for seed in "${seeds[@]}"; do
    for e in "${update_epochs[@]}";do
    for env in "${envS[@]}"; do
        for algo in "${alogs[@]}";do
            /home/weijun.luo/.conda/envs/mujoco_py311/bin/python atari/main.py --env_id ${env} --update_epochs ${e} --seed ${seed} --algo ${algo} --project_name ${project}
        done
    done
    done
done
