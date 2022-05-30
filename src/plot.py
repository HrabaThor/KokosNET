import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_plot(data,save,size=(9,3),show=True):
    '''Plot training info'''
    a_vals, a_ticks = data["a_vals"], data["a_ticks"]
    c_vals, c_ticks = data["c_vals"], data["c_ticks"]
    r_vals, r_ticks = data["r_vals"], data["r_ticks"]
    rewards = data["history_reward"]
    sns.set_theme()

    fig, (ax_a, ax_c, ax_r) = plt.subplots(1, 3, figsize=size)

    sns.lineplot(x=a_ticks, y=a_vals, ax = ax_a, color="red")
    ax_a.set_title("Actor Loss")
    ax_a.set_xlabel("Steps")

    sns.lineplot(x=c_ticks, y=c_vals, ax = ax_c, color="green")
    ax_c.set_title("Critic Loss")
    ax_c.set_xlabel("Steps")

    sns.lineplot(data=rewards, ax = ax_r, color="blue")
    ax_r.set_title("Average Reward")
    ax_r.set_xlabel("Episodes")

    plt.tight_layout(pad=0.5)
    if save:
        fig.savefig(save)
    if show:
        plt.show()


if __name__ == "__main__":
    data = np.load("reacher/train_data"+".npz")
    # print(data["c_vals"])
    create_plot(data,"reacher/test.pdf")