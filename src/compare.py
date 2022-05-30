import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

parser = argparse.ArgumentParser()

<<<<<<< HEAD
parser.add_argument("--env", type=str, 
                    help="Name of environment to compare.")
parser.add_argument("--r-avg", type=int, default=256,
                    help="Choose best model according to mean of N last rewards")
parser.add_argument("--sb3-dir", type=str, 
                    help="Directory with baseline data")
parser.add_argument("--dir", type=str, 
                    help="Directory with our data")
parser.add_argument("--save-dir", type=str, default="comparisons",
                    help="Directory to save the plot into")
# create graphs for report
parser.add_argument("--compare-all", action='store_true',
                    help="Load weights before training")
=======
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
>>>>>>> 6ddcbdc (update repo)

args = parser.parse_args()



def get_sb3_data(dir):
<<<<<<< HEAD
    '''
        Load data from monitor that were collected during the training of stable
        baseline 3 TD3 algorithm.
    '''
=======
>>>>>>> 6ddcbdc (update repo)
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
<<<<<<< HEAD
    '''
        Load data that were collected during the training of our algorithm.
    '''
=======
>>>>>>> 6ddcbdc (update repo)
    # load our data
    data = np.load(dir+"/train_data"+".npz")
    rewards = data["history_reward"]

    return rewards

def plot_rewards(sb3_data, our_data):
<<<<<<< HEAD
    '''
        Create a graph displaying average rewards from both algorithms.
    '''
=======
>>>>>>> 6ddcbdc (update repo)
    sns.set_theme()
   
    # plot both data
    fig = plt.figure(figsize=(10,5))
    sns.lineplot(data=our_data,  color="blue")
    sns.lineplot(data=sb3_data,  color="red")

    fig.suptitle('Average Reward - '+args.env)
    plt.xlabel('Episodes')
    fig.legend(labels=['Our-TD3','SB3-TD3'])

    fig.savefig(args.save_dir+"/"+args.env+"_cmp.pdf")

<<<<<<< HEAD

def plot_all():
    '''
        Create a graph for the purposes of project report containing average
        reward for all three environment.
    '''
    sns.set_theme()    

    fig, (ax_idp, ax_r, ax_hch) = plt.subplots(1, 3, figsize=(9,3))
    axs = [ax_idp, ax_r, ax_hch]
    sb3_dirs = ["sb3_InvertedDoublePendulum","sb3_Reacher","sb3_HalfCheetah"]
    our_dirs = ["InvertedDoublePendulum","Reacher","HalfCheetah"]
    for env, (sb3_dir,our_dir) in enumerate(zip(sb3_dirs,our_dirs)):
        sb3_data = get_sb3_data(sb3_dir)
        our_data = get_our_data(our_dir)

        sns.lineplot(data=sb3_data, ax = axs[env], color="red")
        sns.lineplot(data=our_data, ax = axs[env], color="blue")
        axs[env].set_title(our_dir+"-v2",fontsize=10)
        axs[env].set_xlabel("Episodes",fontsize=8)
        
        
        axs[env].tick_params(axis='both', which='major', labelsize=8)

    axs[1].legend( labels=["SB3-TD3","Our-TD3"],loc='upper center', 
             bbox_to_anchor=(0.5, -0.25),fancybox=False, shadow=False, ncol=3)
    fig.suptitle("Average rewards") 
    plt.tight_layout(pad=0.5)
    fig.savefig("comparison_all.pdf")


if __name__ == "__main__":
    if args.compare_all:
        plot_all()
    else:
        sb3_data = get_sb3_data(args.sb3_dir)
        our_data = get_our_data(args.dir)
        plot_rewards(sb3_data,our_data)
=======
if __name__ == "__main__":
    sb3_data = get_sb3_data(args.sb3_dir)
    our_data = get_our_data(args.dir)
    plot_rewards(sb3_data,our_data)
>>>>>>> 6ddcbdc (update repo)
