import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.replay_buffer import ReplayBuffer
from src.q_values import QValues

class Simulation():
    def __init__(self, env, agent, q_values=None, verb=False, render=False, plot=False, stat=False) -> None:
        self.env = env
        self.agent = agent
        self.q_values = q_values


        self.verb = verb
        self.render = render
        self.plot = plot
        self.stat = stat

        if self.q_values is not None:
            self.buffer = ReplayBuffer(buffer_size=10000, seed=1)
            self.batch_size = 128

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
                action = self.agent.computeAction(state, deterministic=True)
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
            self._plot_reward(step_rewards=step_rewards)

        # print statistics
        if self.stat:
            self._print_stat(step_rewards=step_rewards)

        # plot q-values training loss
        if self.q_values is not None:
            self.q_values.plotLoss()
            self.q_values.plotHeatmap()

            # plot agent training loss (if attribute 'trainStep' is implemented)
            if hasattr(self.agent, 'trainStep') and callable(self.agent.trainStep):
                self.agent.plotLoss()

        return step_rewards
    
    def train(self, num_episodes=10):
        # run episodes
        step_rewards = []
        for i in range(num_episodes):

            # reset environment
            state = self.env.reset()[0] # tuple contains as first element the state
            j = 0
            while True:
                # take action and update environment
                action = self.agent.computeAction(state, deterministic=False)
                next_state, reward, term, trunc, info = self.env.step(action=action)

                # train q-network (if q_values is implemented)
                if self.q_values is not None:
                    # add transition to replay buffer
                    self.buffer.addTransition(state=state, action=action, reward=reward, next_state=next_state, trunc=trunc)
                    batch = self.buffer.sampleBatch(batch_size=self.batch_size)

                    # train q-network if replay buffer is large enough
                    self.q_values.trainStep(batch=batch, agent=self.agent)

                    # train agent (if attribute 'trainStep' is implemented)
                    if hasattr(self.agent, 'trainStep') and callable(self.agent.trainStep):
                        self.agent.trainStep(batch=batch)

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

        # plot rewards
        if self.plot:
            self._plot_reward(step_rewards=step_rewards)

        # print statistics
        if self.stat:
            self._print_stat(step_rewards=step_rewards)

        # plot q-values training loss
        if self.q_values is not None:
            self.q_values.plotLoss()
            self.q_values.plotHeatmap()

            # plot agent training loss (if attribute 'trainStep' is implemented)
            if hasattr(self.agent, 'trainStep') and callable(self.agent.trainStep):
                self.agent.plotLoss()

        return step_rewards
    
    def _print_stat(self, step_rewards):
        # assure that episode length is 200
        assert len(step_rewards) % 200 == 0

        # print statistics
        print(f"Statistics for {int(len(step_rewards) % 200)} episodes")
        print(f"    Mean (of all episodes) cumulative reward: {np.mean(step_rewards)}")
        print(f"    Std (of all episodes) cumulative reward: {np.std(step_rewards)}")

    def _plot_reward(self, step_rewards):
        # assure that episode length is 200
        assert len(step_rewards) % 200 == 0

        # average losses of one episode
        i = 0
        episode_mean = []
        episode_std = []
        while i < len(step_rewards):
            episode_mean.append(np.mean(step_rewards[i:i+200]))
            episode_std.append(np.std(step_rewards[i:i+200]))
            i += 200

        fig = plt.figure()
        plt.errorbar(x=range(len(episode_mean)), y=episode_mean, yerr=episode_std, ecolor="red", label="Reward per episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()
    
