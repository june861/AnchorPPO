update_epochs=(4)
algos=(
    "appo-all"
    "appo-two"
    "ppo"
    # "appo-pow"
    # "spo"
)
envs=(
    "AssaultNoFrameskip-v4"
    "AlienNoFrameskip-v4"
    "AmidarNoFrameskip-v4"
    "AsterixNoFrameskip-v4"
    "AsteroidsNoFrameskip-v4"
    "AtlantisNoFrameskip-v4"
    "BankHeistNoFrameskip-v4"
    "BattleZoneNoFrameskip-v4"
    "BeamRiderNoFrameskip-v4"
    "BerzerkNoFrameskip-v4"
    "BowlingNoFrameskip-v4"
    "BoxingNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "CentipedeNoFrameskip-v4"
    "ChopperCommandNoFrameskip-v4"
    "CrazyClimberNoFrameskip-v4"
    "DemonAttackNoFrameskip-v4"
    "DoubleDunkNoFrameskip-v4"
    "EnduroNoFrameskip-v4"
    "FishingDerbyNoFrameskip-v4"
    "FreewayNoFrameskip-v4"
    "FrostbiteNoFrameskip-v4"
    "GopherNoFrameskip-v4"
    "GravitarNoFrameskip-v4"
    "HeroNoFrameskip-v4"
    "IceHockeyNoFrameskip-v4"
    "JamesbondNoFrameskip-v4"
    "KangarooNoFrameskip-v4"
    "KrullNoFrameskip-v4"
    "KungFuMasterNoFrameskip-v4"
    "MontezumaRevengeNoFrameskip-v4"
    "MsPacmanNoFrameskip-v4"
    "NameThisGameNoFrameskip-v4"
    "PhoenixNoFrameskip-v4"
    "PitfallNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "PrivateEyeNoFrameskip-v4"
    "QbertNoFrameskip-v4"
    "RiverraidNoFrameskip-v4"
    "RoadRunnerNoFrameskip-v4"
    "RobotankNoFrameskip-v4"
    "SeaquestNoFrameskip-v4"
    "SpaceInvadersNoFrameskip-v4"
    "StarGunnerNoFrameskip-v4"
    "TennisNoFrameskip-v4"
    "TimePilotNoFrameskip-v4"
    "TutankhamNoFrameskip-v4"
    "UpNDownNoFrameskip-v4"
    "VentureNoFrameskip-v4"
    "VideoPinballNoFrameskip-v4"
    "WizardOfWorNoFrameskip-v4"
    "YarsRevengeNoFrameskip-v4"
    "ZaxxonNoFrameskip-v4"
)
seeds=(1)
project="atari-v1"
for seed in "${seeds[@]}"; do
    for e in "${update_epochs[@]}"; do
        for env in "${envs[@]}"; do
            for algo in "${algos[@]}"; do
                /home/weijun.luo/.conda/envs/mujoco_py311/bin/python atari/main.py --env_id ${env} --update_epochs ${e} --seed ${seed} --algo ${algo} --project_name ${project}
            done
        done
    done
done
