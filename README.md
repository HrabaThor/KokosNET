# Project KNN - Actor-Critic
### Authors: Martin Kostelník, Michal Glos, Michal Szymik

## Install
To install dependencies:
`pip install -r requirements.txt`

## Examples of runnning our TD3 algorithm
Run training:

`python3 td3.py --plot-show --train 10000 --plot-save plt.pdf --bins=256`

Runs training with 10000 steps, saves losses and rewards as chart into plt.pdf, shows it. Plots all functions as approximation of 256 points (actually creates histogram).

Run eval:

`python3 td3.py --eval --render`

Renders and shows agent performing in an environment

## Examples of running stable-baseline-3 TD3 algorithm

Run training:

`python baselines.py --env HalfCheetah-v2 --dir sb3_HalfCheetah --train --net-arch 400 300`

Run eval:

`python baselines.py --env HalfCheetah-v2 --dir sb3_HalfCheetah --eval`

Run rendering:

`python baselines.py --env HalfCheetah-v2 --dir sb3_HalfCheetah --show`

## Comparing stable-baselines-3 with our algoritm

Plot average rewards:

`python compare.py --env "HalfCheetah-v2" --sb3-dir sb3_halfcheetah --dir HalfCheetah`
