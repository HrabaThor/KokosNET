import pickle
import os
import sys
import base64
from pathlib import Path
from IPython import display as ipythondisplay
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
import numpy as np

class AgentHandle:
    '''
    Helper static class to manipulate with agents.

    Some points:
        Pickling might not be the best practice for naving neural network 
            but it comes handy when you do not want to care about which 
            network class was used and what parameters it took into constructor.

    Provides:
        pickle:     Provide agent as an argument and save it's net (agent.net) as pickle file
        unpickle:   Provide agent as an argument and load it's ner (agent.net) from a pickle file
        load_best:  Loads the best performing neural network from pickle file (agent.best)
        perform:    Show agent performing in gym on video
    '''

    @staticmethod
    def pickle(agent, filename="model", path="models"):
        '''
        Pickle agent's neural network into provided path (path/filename.p)
        
        Argument:
            agent:      Agent class instance
            filename:   Name of file to be the net pickled in (.p extension will be added)
        Returns:
            Path to the pickled file
        '''
        # Make sure directory for storing agents exist
        os.makedirs(path, exist_ok=True)
        # Dump the picklefile
        path = os.path.join(path, filename + ".p")
        with open(path, "wb") as pfile:
            pickle.dump(agent.net, pfile)
        return path

    @staticmethod
    def unpickle(agent, path):
        '''
        Upickle agent's neural network from provided path and set it as it's property

        Arguments:
            path:   Path to neural network pickle file
        '''
        if not os.path.exists(path):
            print(f"File {path} does not exist.", file=sys.stderr)
            return None
        with open(path, "rb") as pfile:
            agent.net =  pickle.load(pfile)

    @staticmethod
    def load_best(agent):
        '''Loads the best performing neural network into provided agent'''
        agent.net = AgentHandle.unpickle(agent, agent.best)

    @staticmethod
    def perform(agent):
        '''Make the agent perform in a environment, show a video of it'''
        # Create the monitor environment from agents initialized environment
        agent.monitor_env = Monitor(
            agent.env, "./videos", force=True, video_callable=lambda episode: True)
        # Let the agent perform
        reward = agent.evaluate(env=agent.monitor_env)
        ######################################### Magic here ###################################################
        # This code was stolen somewhere from the internet by Michal Szimik (hence no reference :( )
        print(f"Reward: {reward}")
        html = []
        for mp4 in Path('./videos').glob("*.mp4"):
            video_b64 = base64.b64encode(mp4.read_bytes())
            html.append('''<video alt="{}" autoplay 
                        loop controls style="height: 400px;">
                        <source src="data:video/mp4;base64,{}" type="video/mp4" />
                    </video>'''.format(mp4, video_b64.decode('ascii')))
        ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


    @staticmethod
    def plot_losses(agent, keys=['total', 'actor', 'critic', 'steps'], rolling=1000):
        '''Plot the development of loss function'''
        # Convolve losses first
        losses = {}
        for key in keys:
            padding = len(agent.net.losses[key]) % rolling
            losses[key] = np.sum(agent.net.losses[key][padding:].reshape((-1, rolling)), axis=1)
        # Create subplots
        fig, axes = plt.subplots(len(keys), figsize=(16,16))
        fig.suptitle("Losses throughout learning")
        for ax, key in zip(axes, keys):
            ax.plot(losses[key])
            ax.set_title(f'{key} loss')
        plt.tight_layout()
        plt.show()
