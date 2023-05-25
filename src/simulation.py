import numpy as np
import pandas as pd
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
        if hasattr(self.actor, "tau"):
            dir_name = dir_name + "_tau"+str(self.actor.tau) + "_theta"+str(self.actor.noise.theta)
        self.dir_path = os.path.join(dir_path, dir_name)
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.mkdir(self.dir_path)

    def run(self, num_episodes, render, plot):
        # create figure for rendering
        if render:
            fig = plt.figure()
            frames = []

        # run episodes
        step_rewards = []
        cum_rewards = []
        for i in range(num_episodes):
            # print testing progress
            if i%10 == 0:
                print(f"Testing episode: {i}/{num_episodes}")

            # reset environment
            state = self.buffer.numpy2tensor(self.env.reset()[0], shape_type="state") # tuple contains as first element the state
            while True:
                # take action and update environment
                action = self.actor.computeActions(states=state, target=False, deterministic=True)
                next_state, reward, term, trunc, info = self.env.step(action=action.detach().numpy().flatten()) # TODO: action has wrond dimensions
                state = self.buffer.numpy2tensor(next_state, shape_type="state")

                # log reward
                step_rewards.append(reward)

                # render environment
                if render:
                    env_screen = self.env.render()
                    frames.append([plt.imshow(env_screen)])       

                # check if episode is truncated
                if trunc:
                    assert len(step_rewards) % 200 == 0 # verfiy that episode length is 200
                    break

            # log cummulative reward
            cum_rewards.append(np.sum(step_rewards[i*200:]))

        # show animation of environment
        if render:
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
            plt.show()

        # plot rewards
        if plot:
            self._plotReward(step_rewards=step_rewards, cum_rewards=cum_rewards, path=self.dir_path, title="reward_testing")

        return step_rewards, cum_rewards
    
    def train(self, num_episodes, batch_size):
        # plot heat maps before training
        self._plotHeatmap(path=self.dir_path, title="heatmap_before_training")
        self._plotPolarHeatMap(path=self.dir_path, title="polar_heatmap_before_training")

        # run episodes
        step_rewards = []
        cum_rewards = []
        for i in range(num_episodes):
            # print training progress
            if i%10 == 0:
                print(f"Training episode: {i}/{num_episodes}")

            # reset environment
            state = self.buffer.numpy2tensor(self.env.reset()[0], shape_type="state") # tuple contains as first element the state
            if hasattr(self.actor, "noise"):
                self.actor.noise.reset()
            while True:
                # take action and update environment
                action = self.actor.computeActions(states=state, target=False, deterministic=False)
                next_state, reward, term, trunc, info = self.env.step(action=action.detach().numpy().flatten())

                # add transition to replay buffer
                self.buffer.addTransition(state=state, action=action, reward=reward, next_state=next_state, trunc=trunc)
                state = self.buffer.numpy2tensor(next_state, shape_type="state")

                # train Q and policy networks if replay buffer is large enough
                batch = self.buffer.sampleBatch(batch_size=batch_size)
                self.critic.trainStep(batch=self.buffer.detachClone(batch), actor=self.actor)
                self.actor.trainStep(batch=self.buffer.detachClone(batch), critic=self.critic)

                # log reward
                step_rewards.append(reward)

                # check if episode is truncated
                if trunc:
                    assert len(step_rewards) % 200 == 0 # verfiy that episode length is 200
                    break

            # log cummulative reward
            cum_rewards.append(np.sum(step_rewards[i*200:]))

        # plot rewards, losses and heat maps
        self._plotReward(step_rewards=step_rewards, cum_rewards=cum_rewards, path=self.dir_path, title="reward_training")
        self._plotLosses(critic_losses=self.critic.log_losses, actor_losses=self.actor.log_losses, path=self.dir_path)
        self._plotHeatmap(path=self.dir_path)
        self._plotPolarHeatMap(path=self.dir_path)

        # save models
        self.critic.saveModels(path=self.dir_path)
        self.actor.saveModels(path=self.dir_path)

        return step_rewards, cum_rewards

    def _plotReward(self, step_rewards, cum_rewards, path, title="reward"):
        # assure that episode length is 200
        assert len(step_rewards) % 200 == 0

        # average losses of one episode
        episode_mean = []
        episode_per5 = []
        episode_per95 = []
        i = 0
        while i < len(step_rewards):
            episodes_rewards = np.array(step_rewards[i:i+200])
            episode_mean.append(episodes_rewards.mean())
            episode_per5.append(np.percentile(episodes_rewards, 5))
            episode_per95.append(np.percentile(episodes_rewards, 95))
            i += 200

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        fig.suptitle("Rewards per episode")

        axs[0].plot(range(len(episode_mean)), cum_rewards, label="Cummulative", color="red")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")
        axs[0].legend()
        axs[0].set_title(f"Cummulative reward per episode (avg={np.round(np.mean(cum_rewards), 3)})")

        axs[1].plot(range(len(episode_mean)), episode_mean, label="Mean", color="red")
        axs[1].fill_between(x=range(len(episode_mean)), y1=episode_per5, y2=episode_per95, alpha=0.2, color="blue", label="Percentile 5-95%")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Reward")
        axs[1].legend()
        axs[1].set_title(f"Mean reward per episode (avg={np.round(np.mean(episode_mean), 3)})")

        plt.savefig(os.path.join(path, title+".pdf"))

    def _plotLosses(self, critic_losses, actor_losses, path):
        # assure that episode length is 200
        assert len(critic_losses) % 200 == 0 and len(critic_losses) == len(actor_losses)
        
        c_losses = []
        a_losses = []
        i = 0
        while i < len(critic_losses):
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

        if hasattr(self.actor, "tau"):
            df = pd.DataFrame({"critic":c_losses, "actor":a_losses, "tau":self.critic.tau, "theta":self.actor.noise.theta})
        else:
            df = pd.DataFrame({"critic":c_losses, "actor":a_losses})
        df.to_csv(os.path.join(path, "losses.csv"))

    def _plotHeatmap(self, path, title="heatmap"):
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

        q_values = []
        for idx, t in enumerate(np.linspace(-1, 1, 9)):
            state_copy = state.detach().clone()

            torque = t * torch.ones(state_copy.shape[0])
            q_val = self.critic.computeQValues(states=state_copy, actions=torque, target=False)
            q_val = q_val.reshape(angle.shape).detach().numpy()
            q_values.append(q_val)

        q_val_max = np.max(q_values)
        q_val_min = np.min(q_values)

        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
        fig.suptitle('Estimated V-values')
        for idx, q_val in enumerate(q_values):
            i = idx // 3
            j = idx % 3

            colorbar = axs[i,j].pcolormesh(angle, vel, q_val, vmin=q_val_min, vmax=q_val_max)
            axs[i,j].axis([np.min(angle), np.max(angle), np.min(vel), np.max(vel)])
            fig.colorbar(colorbar, ax=axs[i,j])
            axs[i,j].set_xlabel("Angle [rad]")
            axs[i,j].set_ylabel("Velocity [rad/s]")
            axs[i,j].set_title(f"Torque = {2*t}Nm")

        plt.savefig(os.path.join(path, title+".pdf"))

    def _plotPolarHeatMap(self, path, title="polar_heatmap"):
        res_angle = 360
        res_radial = 10

        radius = np.linspace(0, 1, res_radial)
        angle = np.linspace(-np.pi, np.pi, res_angle)

        r, a = np.meshgrid(radius, angle)
        cos_a = torch.cos(torch.tensor(a, dtype=torch.float32)).reshape(-1,1)
        sin_a = torch.sin(torch.tensor(a, dtype=torch.float32)).reshape(-1,1)   

        velocities = [0, 2.5]
        torques = [-1, 0, 1]
        q_values = []
        for i, v in enumerate(velocities):
            for j, torque in enumerate(torques):
                vel = v * torch.ones_like(cos_a, dtype=torch.float32)
                states = torch.concat((cos_a, sin_a, vel), axis=1)
                actions = torque * torch.ones_like(cos_a, dtype=torch.float32)
                q_val = self.critic.computeQValues(states=states, actions=actions.reshape(-1,1), target=False)

                q_val = q_val.detach().numpy().reshape(res_angle, res_radial)
                q_values.append(q_val)
                
        
        q_val_max = np.max(q_values)
        q_val_min = np.min(q_values)

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8.5,12), subplot_kw={'projection':"polar"})      
        for idx, q_val in enumerate(q_values):
            i = idx // len(torques)
            j = idx % len(torques)
                    
            cb = axs[j,i].pcolormesh(a, r, q_val, vmin=q_val_min, vmax=q_val_max)
            axs[j,i].plot(angle, r, color='k', ls='none') 
            fig.colorbar(cb, ax=axs[j,i])
            axs[j,i].set_yticks([],[])
            axs[j,i].set_theta_offset(np.pi/2)
            axs[j,i].set_theta_direction(-1)
            axs[j,i].set_title(f"Torque={2*torques[j]}Nm, vel={velocities[i]}m/s")

        plt.savefig(os.path.join(path, title+".pdf"))

    
