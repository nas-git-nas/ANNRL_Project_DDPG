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
from src.replay_buffer import ReplayBuffer

class Simulation():
    def __init__(
            self,
            dir_path:str, 
            env:NormalizedEnv, 
            critic:Critic,
            actor:Actor,
            buffer:ReplayBuffer,
        ) -> None:

        self.env = env
        self.critic = critic
        self.actor = actor
        self.buffer = buffer

        # create directory
        t = datetime.now()
        dir_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M")
        self.dir_path = os.path.join(dir_path, dir_name)
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.mkdir(self.dir_path)

    def run(self, num_episodes, render):
        # create figure for rendering
        if render:
            fig = plt.figure()
            frames = []

        # run episodes
        step_rewards = []
        for i in range(num_episodes):

            # reset environment
            state = self.buffer.numpy2tensor(self.env.reset()[0]) # tuple contains as first element the state
            while True:
                # take action and update environment
                action = self.actor.computeActions(states=state, target=False, deterministic=True)
                next_state, reward, term, trunc, info = self.env.step(action=action.detach().numpy()) # TODO: action has wrond dimensions
                state = self.buffer.numpy2tensor(next_state)

                # log reward
                step_rewards.append(reward)

                # render environment
                if render:
                    env_screen = self.env.render()
                    frames.append([plt.imshow(env_screen)])       

                # check if episode is terminated or truncated
                if term or trunc:
                    break


        # show animation of environment
        if render:
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
            plt.show()

        # plot rewards
        self._plotReward(step_rewards=step_rewards, path=self.dir_path)

        return step_rewards
    
    def train(self, num_episodes, batch_size):
        # run episodes
        step_rewards = []
        for i in range(num_episodes):
            if i%10 == 0:
                print(f"Training episode: {i}/{num_episodes}")

            # reset environment
            state = self.buffer.numpy2tensor(self.env.reset()[0]) # tuple contains as first element the state
            if hasattr(self.actor, "noise"):
                self.actor.noise.reset()
            while True:
                # take action and update environment
                action = self.actor.computeActions(state=state, target=False, deterministic=False)
                next_state, reward, term, trunc, info = self.env.step(action=action.detach().numpy())

                # add transition to replay buffer
                self.buffer.addTransition(state=state, action=action, reward=reward, next_state=next_state, trunc=trunc)
                state = self.buffer.numpy2tensor(next_state)

                # train Q and policy networks if replay buffer is large enough
                batch = self.buffer.sampleBatch(batch_size=batch_size)
                self.critic.trainStep(batch=self.buffer.detachClone(batch), actor=self.actor)
                self.actor.trainStep(batch=self.buffer.detachClone(batch), critic=self.critic)

                # log reward
                step_rewards.append(reward)

                # check if episode is terminated or truncated
                if term or trunc:
                    break

        # plot rewards, losses and heat maps
        self._plotReward(step_rewards=step_rewards, path=self.dir_path)
        self._plotLosses(critic_losses=self.critic.log_losses, actor_losses=self.actor.log_losses, path=self.dir_path)
        self._plotHeatmap(path=self.dir_path)

        # save models
        self.critic.saveModels(path=self.dir_path)
        self.actor.saveModels(path=self.dir_path)

        return step_rewards

    def _plotReward(self, step_rewards, path):
        # assure that episode length is 200
        assert len(step_rewards) % 200 == 0

        # average losses of one episode
        episode_sum = []
        episode_mean = []
        episode_per5 = []
        episode_per95 = []
        i = 0
        while i < len(step_rewards):
            episodes_rewards = np.array(step_rewards[i:i+200])
            episode_sum.append(episodes_rewards.sum())
            episode_mean.append(episodes_rewards.mean())
            episode_per5.append(np.percentile(episodes_rewards, 5))
            episode_per95.append(np.percentile(episodes_rewards, 95))
            i += 200

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        fig.suptitle("Rewards per episode")

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

    def _plotLosses(self, critic_losses, actor_losses, path):
        # assure that episode length is 200
        assert len(critic_losses) % 200 == 0 and len(critic_losses) == len(actor_losses)
        
        c_losses = []
        a_losses = []
        i = 0
        while i < len(self.critic_losses):
            c_losses.append(np.mean(critic_losses[i:i+200]))
            a_losses.append(np.mean(actor_losses[i:i+200]))
            i += 200

        fig = plt.figure()
        plt.plot(c_losses, label="Critic Loss", color="green")
        plt.plot(a_losses, label="Actor Loss", color="blue")
        plt.title("Losses")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(path, "losses.pdf"))

    def _plotHeatmap(self, path):
        res = 20

        angle = torch.linspace(-np.pi, np.pi, res)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        vel = torch.linspace(-5, 5, res)

        meshgrid = torch.meshgrid(torch.arange(res), torch.arange(res))
        idxs_a = meshgrid[0].flatten()
        idxs_vel = meshgrid[1].flatten()

        angle = angle[idxs_a]
        cos_a = cos_a[idxs_a]
        sin_a = sin_a[idxs_a]
        vel = vel[idxs_vel]

        state = torch.concat((cos_a.reshape(-1,1), sin_a.reshape(-1,1), vel.reshape(-1,1)), axis=1)
        angle = angle.reshape(meshgrid[0].shape).detach().numpy()
        vel = vel.reshape(angle.shape).detach().numpy()

        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
        fig.suptitle('Estimated V-values')
        for idx, t in enumerate(np.linspace(-1, 1, 9)):
            i = idx // 3
            j = idx % 3
            state_copy = state.detach().clone()

            torque = t * torch.ones(state_copy.shape[0])
            v_val = self.critic.computeQValues(states=state_copy, actions=torque, target=False)
            v_val = v_val.reshape(angle.shape).detach().numpy()
            
            colorbar = axs[i,j].pcolormesh(angle, vel, v_val)
            axs[i,j].axis([np.min(angle), np.max(angle), np.min(vel), np.max(vel)])
            fig.colorbar(colorbar, ax=axs[i,j])
            axs[i,j].set_xlabel("Angle [rad]")
            axs[i,j].set_ylabel("Velocity [rad/s]")
            axs[i,j].set_title(f"Torque = {2*t}Nm")

        plt.savefig(os.path.join(path, "heatmap.pdf"))
    
