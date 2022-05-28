import argparse
from agent import Agent

# Init parser
parser = argparse.ArgumentParser()
# Add all possible options
# Coefficients:
parser.add_argument("--noise", type=float, default=0.1,
                    help="Gaussian noise std for sampling actions when training")
parser.add_argument("--lra", type=float, default=0.001,
                    help="Learning rate of agent")
parser.add_argument("--lrc", type=float, default=0.001,
                    help="Learning rate for critics")
parser.add_argument("--tau", type=float, default=0.005,
                    help="Coefficient for morphind nets and it's targets")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="Gamma coefficients for calculating discounts")
# Replay buffer
parser.add_argument("--b-size", type=int, default=1000000,
                    help="Size of replay buffer")
# Training
parser.add_argument("--warmup", type=int, default=100000,
                    help="Populate buffer before training with warmup steps")
parser.add_argument("--train", type=int, default=100000,
                    help="Train the model for N steps")
parser.add_argument("--policy-delay", type=int, default=2,
                    help="Learn actors each N learning steps")
parser.add_argument("--pbar-update", type=int, default=5,
                    help="Update the progress bar eeach N finished games")
parser.add_argument("--r-avg", type=int, default=256,
                    help="Choose best model according to mean of N last rewards")
parser.add_argument("--batch", type=int, default=64,
                    help="For each env step, perforl learning from BATCH samples")
# Evaluation
parser.add_argument("--eval", action='store_true',
                    help="Evaluate model")
parser.add_argument("--epochs", type=int, default=24,
                    help="Epochs for evaluating model")
parser.add_argument("--render", action='store_true',
                    help="Render evaluation process")
parser.add_argument("--fps", type=float, default=float("inf"),
                    help="FPS when rendering evaluation")
# Plotting
parser.add_argument("--plot-show", action='store_true',
                    help="Show plotted losses and rewards")
parser.add_argument("--plot-save", type=str, default="",
                    help="Save plotted losses and rewards")
parser.add_argument("--bins", type=int, default=128,
                    help="Plot values as B points")
parser.add_argument("--figsize", type=int, nargs=2, default=(9,10),
                    help="Define size of plotted chart")
# Agent config
parser.add_argument("--critic", type=int, nargs='+',
                    default=[400, 300], help="Specify layers of critic")
parser.add_argument("--actor", type=int, nargs='+',
                    default=[400, 300], help="Specify layers of actor")
parser.add_argument("--load", action='store_true',
                    help="Load weights before training")
# Paths
parser.add_argument("--save", type=str, default="model",
                    help="Directory for saving models")
# Environment
parser.add_argument("--max-env-steps", type=int, default=200,
                    help="Max steps in environment before resetting")
parser.add_argument("--env", type=str, default="Pendulum-v1",
                    help="Set environmnet ID")

# Parse arguments
args = parser.parse_args()

if __name__ == "__main__":
    # Initialize the agent
    agent = Agent(env_id=args.env, noise=args.noise, buffer=args.b_size,
                  warmup=0 if args.eval else args.warmup, model_dir=args.save, lra=args.lra,
                  lrc=args.lrc, tau=args.tau, gamma=args.gamma,
                  a_layers=args.actor, c_layers=args.critic, load=args.load)
    # If evaluation is requested, load the model and evaluate it
    if args.eval:
        agent.load()
        agent.evaluate(epochs=args.epochs, render=args.render, fps=args.fps)
    # Else - train the model
    else:
        # Try - when killed by ctrl + c - plot it nevertheless
        try:
            agent.train(args.train, batch_size=args.batch, policy_delay=args.policy_delay,
                        r_avg=args.r_avg, max_steps=args.max_env_steps, pbar_update=args.pbar_update)
        finally:
            # Plot losses and rewards
            if args.plot_show or args.plot_save:
                agent.plot(args.plot_save, args.plot_show, args.figsize, args.bins)