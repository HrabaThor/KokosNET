import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, required=True,
                    help="Name of environment to compare.")
parser.add_argument("--r-avg", type=int, default=256,
                    help="Choose best model according to mean of N last rewards")
parser.add_argument("--sb3-dir", type=str, required=True,
                    help="Directory with baseline data")
parser.add_argument("--dir", type=str, required=True,
                    help="Directory with our data")
parser.add_argument("--save-dir", type=str, default="comparisons",
                    help="Directory to save the plot into")

args = parser.parse_args()



def get_sb3_data(dir):
    # load baseline data
    e, r = ts2xy(load_results(dir), 'episodes')
    rewards = np.array(r)
    tmp = np.empty(0)
    data = []
    for i in range(len(rewards)):
        tmp = np.append(tmp,rewards[i])
        data = np.append(data, tmp[-args.r_avg:].mean())
    return data

def get_our_data(dir):
    # load our data
    data = np.load(dir+"/train_data"+".npz")
    rewards = data["history_reward"]

    return rewards

def plot_rewards(sb3_data, our_data):
    sns.set_theme()
   
    # plot both data
    fig = plt.figure(figsize=(10,5))
    sns.lineplot(data=our_data,  color="blue")
    sns.lineplot(data=sb3_data,  color="red")

    fig.suptitle('Average Reward - '+args.env)
    plt.xlabel('Episodes')
    fig.legend(labels=['Our-TD3','SB3-TD3'])

    fig.savefig(args.save_dir+"/"+args.env+"_cmp.pdf")

if __name__ == "__main__":
    sb3_data = get_sb3_data(args.sb3_dir)
    our_data = get_our_data(args.dir)
    plot_rewards(sb3_data,our_data)