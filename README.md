To install dependencies:
`pip install -r requirements.txt`

Run training:

`python3 td3.py --plot-show --train 10000 --plot-save plt.pdf --bins=256`

Runs training with 10000 steps, saves losses and rewards as chart into plt.pdf, shows it. Plots all functions as approximation of 256 points (actually creates histogram).

Run eval:

`python3 td3.py --eval --render`

Renders and shows agent performing in an environment