import numpy as np
import torch
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.replay_buffer import ReplayBuffer
from src.environment import NormalizedEnv
from src.critic import Critic
from src.actor import Actor

class Simulation():
    def __init__(self, buffer_size, dir_path:str, env:NormalizedEnv, actor:Actor, critic:Critic=None, verb=False, render=False, plot=False, stat=False) -> None:
        self.env = env
        self.actor = actor
        self.critic = critic

        self.verb = verb
        self.render = render
        self.plot = plot
        self.stat = stat

        if self.critic is not None:
            self.buffer = ReplayBuffer(buffer_size=buffer_size, seed=1)
            self.batch_size = 128

        # create directory
        t = datetime.now()
        dir_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M")
        self.dir_path = os.path.join(dir_path, dir_name)
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.mkdir(self.dir_path)

    def run(self, num_episodes=10):
        # create figure for rendering
        if self.render:
            fig = plt.figure()
            frames = []

        # run episodes
        step_rewards = []
        for i in range(num_episodes):

            # reset environment
            state = self.env.reset()[0] # tuple contains as first element the state
            j = 0
            while True:
                # take action and update environment
                action = self.actor.computeAction(state, deterministic=True, use_target_network=False)
                next_state, reward, term, trunc, info = self.env.step(action=action)             
                
                # render environment
                if self.render:
                    env_screen = self.env.render()
                    frames.append([plt.imshow(env_screen)])

                # print info
                if self.verb:
                    print(f"step: {j}, action: {action}, reward: {reward}, terminated: {term}, truncated: {trunc}, info: {info}")

                # update state
                state = next_state
                step_rewards.append(reward)
                j += 1

                # check if episode is terminated or truncated
                if term:
                    if self.verb:
                        print(f"Episode terminated, steps: {j}")
                    break
                if trunc:
                    if self.verb:
                        print(f"Episode truncated, steps: {j}")
                    break


        # show animation of environment
        if self.render:
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
            plt.show()

        # plot rewards
        if self.plot:
            self._plot_reward(step_rewards=step_rewards, path=self.dir_path)

        # print statistics
        if self.stat:
            self._print_stat(step_rewards=step_rewards)

        # plot q-values training loss
        if self.critic is not None:
            self.critic.plotLoss(path=self.dir_path)
            self.critic.plotHeatmap(path=self.dir_path)

            # plot actor training loss (if attribute 'trainStep' is implemented)
            if hasattr(self.actor, 'trainStep') and callable(self.actor.trainStep):
                self.actor.plotLoss(path=self.dir_path)

        return step_rewards
    
    def train(self, num_episodes=10):
        # run episodes
        step_rewards = []
        for i in range(num_episodes):
            if i%10 == 0:
                print(f"Training episode: {i}/{num_episodes}")

            # reset environment
            state = self.buffer.numpy2tensor(self.env.reset()[0]) # tuple contains as first element the state
            j = 0
            while True:
                # take action and update environment
                action = self.actor.computeAction(state, deterministic=False, use_target_network=False)
                next_state, reward, term, trunc, info = self.env.step(action=action.detach().numpy())

                # train q-network (if critic is implemented)
                if self.critic is not None:
                    # add transition to replay buffer
                    self.buffer.addTransition(state=state, action=action, reward=reward, next_state=next_state, trunc=trunc)
                    batch = self.buffer.sampleBatch(batch_size=self.batch_size)

                    # train q-network if replay buffer is large enough
                    self.critic.trainStep(batch=self.buffer.detachClone(batch), actor=self.actor)

                    # train actor
                    self.actor.trainStep(batch=self.buffer.detachClone(batch))

                # update state
                state = self.buffer.numpy2tensor(next_state)
                step_rewards.append(reward)
                j += 1

                # check if episode is terminated or truncated
                if term:
                    if self.verb:
                        print(f"Episode terminated, steps: {j}")
                    break
                if trunc:
                    if self.verb:
                        print(f"Episode truncated, steps: {j}")
                    break

        # plot rewards
        if self.plot:
            self._plot_reward(step_rewards=step_rewards, path=self.dir_path)

        # print statistics
        if self.stat:
            self._print_stat(step_rewards=step_rewards)

        # plot q-values training loss
        if self.critic is not None:
            self.critic.plotLoss(path=self.dir_path)
            self.critic.plotHeatmap(path=self.dir_path)

            # plot actor training loss (if attribute 'trainStep' is implemented)
            if hasattr(self.actor, 'trainStep') and callable(self.actor.trainStep):
                self.actor.plotLoss(path=self.dir_path)

        return step_rewards
    
    def _print_stat(self, step_rewards):
        # assure that episode length is 200
        assert len(step_rewards) % 200 == 0

        # print statistics
        print(f"Statistics for {int(len(step_rewards) % 200)} episodes")
        print(f"    Mean (of all episodes) cumulative reward: {np.mean(step_rewards)}")
        print(f"    Std (of all episodes) cumulative reward: {np.std(step_rewards)}")

    def _plot_reward(self, step_rewards, path):
        # assure that episode length is 200
        assert len(step_rewards) % 200 == 0

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        fig.suptitle("Cummulativ reward per episode")

        # average losses of one episode
        episode_mean = np.empty((len(step_rewards)//200))
        episode_std = np.empty((len(step_rewards)//200))
        for i in range(len(step_rewards)//200):
            episode_mean[i] = np.mean(step_rewards[i:i+200])
            episode_std[i] = np.std(step_rewards[i:i+200])

        axs[0].plot(range(len(episode_mean)), episode_mean, label="Mean", color="red")
        axs[0].fill_between(x=range(len(episode_mean)), y1=episode_mean-episode_std, y2=episode_mean+episode_std, alpha=0.2, color="blue", label="Std")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")
        axs[0].legend()
        axs[0].set_title("Reward per episode")


        # average losses of 1000 steps
        episode_mean = np.empty((len(step_rewards)//1000))
        episode_std = np.empty((len(step_rewards)//1000))
        for i in range(len(step_rewards)//1000):
            episode_mean[i] = np.mean(step_rewards[i:i+1000])
            episode_std[i] = np.std(step_rewards[i:i+1000])

        axs[1].plot(range(len(episode_mean)), episode_mean, label="Mean", color="red")
        axs[1].fill_between(x=range(len(episode_mean)), y1=episode_mean-episode_std, y2=episode_mean+episode_std, alpha=0.2, color="blue", label="Std")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Reward")
        axs[1].legend()
        axs[1].set_title("Reward per 1000 steps")

        plt.savefig(os.path.join(path, "reward.pdf"))
    
