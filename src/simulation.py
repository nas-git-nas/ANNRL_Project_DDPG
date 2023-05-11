import numpy as np
import torch
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.replay_buffer import ReplayBuffer
from src.environment import NormalizedEnv
from src.critic_actor import CriticActor
from src.gaussian_action_noise import GaussianActionNoise
from src.replay_buffer import ReplayBuffer

class Simulation():
    def __init__(
            self,
            dir_path:str, 
            env:NormalizedEnv, 
            critic_actor:CriticActor, 
            noise:GaussianActionNoise,
            buffer:ReplayBuffer,
        ) -> None:

        self.env = env
        self.critic_actor = critic_actor
        self.noise = noise
        self.buffer = buffer

        # create directory
        t = datetime.now()
        dir_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M")
        self.dir_path = os.path.join(dir_path, dir_name)
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.mkdir(self.dir_path)
    
    def train(self, num_episodes, batch_size):
        # run episodes
        step_rewards = []
        for i in range(num_episodes):
            if i%10 == 0:
                print(f"Training episode: {i}/{num_episodes}")

            # reset environment
            state = self.buffer.numpy2tensor(self.env.reset()[0]) # tuple contains as first element the state
            self.noise.reset()
            while True:
                # take action and update environment
                action = self.critic_actor.actor_net(state)
                action = self.noise.getNoisyAction(action)
                next_state, reward, term, trunc, info = self.env.step(action=action.detach().numpy())

                # add transition to replay buffer
                self.buffer.addTransition(state=state, action=action, reward=reward, next_state=next_state, trunc=trunc)
                state = self.buffer.numpy2tensor(next_state)

                # train Q and policy networks if replay buffer is large enough
                batch = self.buffer.sampleBatch(batch_size=batch_size)
                self.critic_actor.trainStep(batch=self.buffer.detachClone(batch))

                # update state
                step_rewards.append(reward)

                # check if episode is terminated or truncated
                if term or trunc:
                    break

        # plot rewards, losses and heat maps
        self._plot_reward(step_rewards=step_rewards, path=self.dir_path)
        self.critic_actor.plotLosses(path=self.dir_path)
        self.critic_actor.plotHeatmap(path=self.dir_path)

        # save models
        self.critic_actor.saveModels(path=self.dir_path)

        return step_rewards

    def _plot_reward(self, step_rewards, path):
        # assure that episode length is 200
        assert len(step_rewards) % 200 == 0

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        fig.suptitle("Rewards per episode")

        # average losses of one episode
        episode_sum = np.zeros((len(step_rewards)//200))
        episode_mean = np.zeros((len(step_rewards)//200))
        episode_per5 = np.zeros((len(step_rewards)//200))
        episode_per95 = np.zeros((len(step_rewards)//200))
        for i in range(len(step_rewards)//200):
            episodes_rewards = np.array(step_rewards[i*200:(i+1)*200])
            episode_sum[i] = episodes_rewards.sum()
            episode_mean[i] = episodes_rewards.mean()
            episode_per5[i] = np.percentile(episodes_rewards, 5)
            episode_per95[i] = np.percentile(episodes_rewards, 95)

        axs[0].plot(range(len(episode_mean)), episode_sum, label="Cummulative", color="red")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")
        axs[0].legend()
        axs[0].set_title("Cummulative reward per episode")

        axs[1].plot(range(len(episode_mean)), episode_mean, label="Mean", color="red")
        axs[1].fill_between(x=range(len(episode_mean)), y1=episode_per5, y2=episode_per95, alpha=0.2, color="blue", label="Percentile 5-95%")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Reward")
        axs[1].legend()
        axs[1].set_title("Mean reward per episode")

        plt.savefig(os.path.join(path, "reward.pdf"))
    
