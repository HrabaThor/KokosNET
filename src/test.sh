#!/bin/bash 

# python3 td3.py --plot-show --train 50000 --plot-save plt.pdf --bins=256 --save mountain_car --env MountainCarContinuous-v0 --gamma 0.99 --critic 150 64 64 --actor 150 64 64 --b-size 5000000 --r-avg 500 --max-env-steps 200 --batch 128 --noise 0.1
# python3 td3.py --eval --render --save mountain_car --epochs 1 --env MountainCarContinuous-v0 --gamma 0.9 --critic 150 64 64 --actor 150 64 64 --policy-delay 4 --r-avg 500 --max-env-steps 100 --batch 128 --noise 0.5


#humanoid_standup

# MUJOCO_ENV="HumanoidStandup-v2"
# MODEL_DIR="humanoid_standup"
# GRAPHS="humanoid_standup.pdf"
# ACTOR="300 150 150 64 64 32"
# CRITIC=$ACTOR
# MAX_STEPS=200
# NOISE=0.1
# GAMMA=0.99
# MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE} --gamma ${GAMMA}"


# if [ $1 == "train" ]
# then
#     python3 td3.py --warmup 10000 --plot-show --train 500000 --plot-save $GRAPHS --bins=256 $MODEL_PARAMS --load
# elif [ $1 == "eval" ]
# then
#     python3 td3.py --eval --render --epochs 20 $MODEL_PARAMS  
# fi

#double inverted pendulum

# MUJOCO_ENV="InvertedDoublePendulum-v2"
# MODEL_DIR="InvertedDoublePendulum"
# GRAPHS="InvertedDoublePendulum.pdf"
# ACTOR="64 32 32 16"
# CRITIC=$ACTOR
# MAX_STEPS=1000
# NOISE=0.1
# MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE} --actor ${ACTOR} --critic ${CRITIC} "


# if [ $1 == "train" ]
# then
#     python3 td3.py --plot-show --train 50000 --plot-save $GRAPHS --bins=256 $MODEL_PARAMS
# elif [ $1 == "eval" ]
# then
#     python3 td3.py --eval --render --epochs 20 $MODEL_PARAMS  
# fi


# Reacher-v2
# MUJOCO_ENV="Reacher-v2"
# MODEL_DIR="reacher"
# GRAPHS="reacher.pdf"
# ACTOR="600 300 100 64"
# CRITIC=$ACTOR
# MAX_STEPS=1000
# NOISE=0.1
# MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE} --actor ${ACTOR} --critic ${CRITIC} "


# if [ $1 == "train" ]
# then
#     python3 td3.py --plot-show --train 50000 --plot-save $GRAPHS --bins=256 $MODEL_PARAMS
# elif [ $1 == "eval" ]
# then
#     python3 td3.py --eval --render --epochs 20 $MODEL_PARAMS  
# fi

# Pendulum-v1
# MUJOCO_ENV="Pendulum-v1"
# MODEL_DIR="pendulum"
# GRAPHS="pendulum.pdf"
# ACTOR="600 300 150"
# CRITIC=$ACTOR
# MAX_STEPS=1000
# NOISE=0.1
# MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE}" # --actor ${ACTOR} --critic ${CRITIC} "


# if [ $1 == "train" ]
# then
#     python3 td3.py --plot-show --train 1000 --plot-save $GRAPHS --bins=256 $MODEL_PARAMS --warmup=100
# elif [ $1 == "eval" ]
# then
#     python3 td3.py --eval --render --epochs 20 $MODEL_PARAMS  
# fi


# HalfCheetah-v2
MUJOCO_ENV="HalfCheetah-v2"
MODEL_DIR="HalfCheetah"
GRAPHS="HalfCheetah.pdf"
ACTOR="600 300 100 64"
CRITIC=$ACTOR
MAX_STEPS=1000
NOISE=0.1
MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE}" # --actor ${ACTOR} --critic ${CRITIC} "


if [ $1 == "train" ]
then
    python3 td3.py --plot-show --train 50000 --plot-save $GRAPHS --bins=256 $MODEL_PARAMS
elif [ $1 == "eval" ]
then
    python3 td3.py --eval --render --epochs 20 $MODEL_PARAMS  
fi