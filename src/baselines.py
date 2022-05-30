import os
import gym
import torch as th
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# Init parser
parser = argparse.ArgumentParser()
parser.add_argument("--load", action='store_true',
                    help="Load weights before training")
parser.add_argument("--plot", action='store_true',
                    help="Load weights before training")
parser.add_argument("--eval", action='store_true',
                    help="Load weights before training")
parser.add_argument("--show", action='store_true',
                    help="Load weights before training")
parser.add_argument("--train", action='store_true',
                    help="Load weights before training")
parser.add_argument("--env", type=str, required=True,
                    help="Set environmnet ID")

parser.add_argument("--r-avg", type=int, default=256,
                    help="Choose best model according to mean of N last rewards")
parser.add_argument("--noise", type=float, default=0.1,
                    help="Gaussian noise std for sampling actions when training")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate of agent")
parser.add_argument("--tau", type=float, default=0.005,
                    help="Coefficient for morphind nets and it's targets")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="Gamma coefficients for calculating discounts")
# Replay buffer
parser.add_argument("--b-size", type=int, default=1000000,
                    help="Size of replay buffer")
parser.add_argument("--batch", type=int, default=64,
                    help="For each env step, perforl learning from BATCH samples")
parser.add_argument("--policy-delay", type=int, default=2,
                    help="Learn actors each N learning steps")
parser.add_argument("--net-arch", type=int, nargs='+',
                    default=[400, 300], help="Specify layers of critic")

parser.add_argument("--dir", type=str, default="sb3_test",
                    help="Directory for saving models")

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
        Taken from https://stable-baselines3.readthedocs.io/en/master/guide/examples.html.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-args.r_avg:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def plot_avg_reward(dir):
    '''
        Create a graph displaying average reward from data collected during
        training.
    '''
    sns.set_theme()
    _, r = ts2xy(load_results(dir), 'episodes')
    rewards = np.array(r)
    tmp = np.empty(0)
    reward_avg = []
    for i in range(len(rewards)):
        tmp = np.append(tmp,rewards[i])
        reward_avg = np.append(reward_avg, tmp[-args.r_avg:].mean())

    fig = plt.figure(figsize=(3,3))
    sns.lineplot(data=reward_avg, color="blue")
    fig.suptitle("Average Reward")
    plt.xlabel("Episodes")
    
    fig.savefig(os.path.join(dir,"plot.pdf"))

def baselines_td3_train(env,dir, load=False):
    '''
        Training of stable baselines 3 algorithm TD3.
    '''
    if load:
        model = TD3.load(os.path.join(dir,"best_model.zip"),env)
    else:

        
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.noise * np.ones(n_actions))
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=args.net_arch)
        model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,policy_kwargs=policy_kwargs,
                    learning_rate=args.lr,gamma=args.gamma,batch_size=args.batch,tau=args.tau,policy_delay=args.policy_delay)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=dir)
    model.learn(total_timesteps=50000,log_interval=1000, callback=callback)

    env.close()

def show_video(dir):
    '''
        Infinite rendering of agent solving current environment.
    '''
    model = TD3.load(os.path.join(dir,"best_model.zip"),env)

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        env.render()

        if done:
            obs = env.reset()

def baseline_td3_eval(dir,episodes=1000):
    '''
        Get mean average reward from model evaluation.
    '''
    model = TD3.load(os.path.join(dir,"best_model.zip"),env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=episodes)
    print("Mean reward: {}, Standard deviation: {}".format(mean_reward,std_reward))

    

# Parse arguments
args = parser.parse_args()

if __name__ == "__main__":
    env = gym.make(args.env)
    
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    if args.train:
        env = Monitor(env, os.path.join(args.dir,"monitor.csv"))
        baselines_td3_train(env,args.dir, load=args.load)
    elif args.eval:
        baseline_td3_eval(args.dir,episodes=1000)
    elif args.plot:
        plot_avg_reward(args.dir) 
    elif args.show:
        show_video(args.dir)

    