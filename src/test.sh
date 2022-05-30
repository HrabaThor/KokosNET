#!/bin/bash 



inv_dbl_pnd () {
    MUJOCO_ENV="InvertedDoublePendulum-v2"
    MODEL_DIR="InvertedDoublePendulum"
    GRAPHS="InvertedDoublePendulum.pdf"
    ACTOR="64 32 32 16"
    CRITIC=$ACTOR
    MAX_STEPS=1000
    NOISE=0.1
    MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE} --actor ${ACTOR} --critic ${CRITIC} "

}


reacher() {
    MUJOCO_ENV="Reacher-v2"
    MODEL_DIR="Reacher"
    GRAPHS="reacher.pdf"
    ACTOR="600 300 100 64"
    CRITIC=$ACTOR
    MAX_STEPS=1000
    NOISE=0.1
    MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE} --actor ${ACTOR} --critic ${CRITIC} "

}

hlf_chth() {
    # HalfCheetah-v2
    MUJOCO_ENV="HalfCheetah-v2"
    MODEL_DIR="HalfCheetah"
    GRAPHS="HalfCheetah.pdf"
    ACTOR="600 300 100 64"
    CRITIC=$ACTOR
    MAX_STEPS=1000
    NOISE=0.1
    MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE}" # --actor ${ACTOR} --critic ${CRITIC} "

}

test() {
    MUJOCO_ENV="Pendulum-v1"
    MODEL_DIR="Pendulum"
    GRAPHS="pendulum.pdf"
    ACTOR="64 32 16"
    CRITIC=$ACTOR
    MAX_STEPS=1000
    NOISE=0.1
    MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE} --actor ${ACTOR} --critic ${CRITIC} "
}

ENV=$1

case $ENV in

    InvertedDoublePendulum-v2)
        inv_dbl_pnd
        ;;
    
    HalfCheetah-v2)
        reacher
        ;;
    
    Reacher-v2)
        hlf_chth
        ;;

    Test)
        test
        ;;
    *)
        echo -n "unknown"
        exit
        ;;
esac

if [ $2 == "train" ]
then
    python3 td3.py --plot-show --train 50000 --plot-save $GRAPHS --bins=256 $MODEL_PARAMS
elif [ $2 == "eval" ]
then
    python3 td3.py --eval --epochs 1000 $MODEL_PARAMS  
fi



# humanoid_standup

# MUJOCO_ENV="HumanoidStandup-v2"
# MODEL_DIR="humanoid_standup"
# GRAPHS="humanoid_standup.pdf"
# ACTOR="300 150 150 64 64 32"
# CRITIC=$ACTOR
# MAX_STEPS=200
# NOISE=0.1
# GAMMA=0.99
# MODEL_PARAMS="--save ${MODEL_DIR} --env ${MUJOCO_ENV} --max-env-steps ${MAX_STEPS} --noise ${NOISE} --gamma ${GAMMA}"

