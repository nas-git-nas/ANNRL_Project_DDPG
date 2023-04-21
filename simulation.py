import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from replay_buffer import ReplayBuffer
from q_values import QValues

class Simulation():
    def __init__(self, env, agent, learn_qval=False, verb=False, render=False, plot=False, stat=False) -> None:
        self.env = env
        self.agent = agent

        self.learn_qval = learn_qval

        self.verb = verb
        self.render = render
        self.plot = plot
        self.stat = stat

        if self.learn_qval:
            self.buffer = ReplayBuffer(max_size=10000, seed=1)
            self.q_values = QValues(gamma=0.99, lr=1e-4)
            self.batch_size = 128

    def run(self, num_episodes=10):
        # create figure for rendering
        if self.render:
            fig = plt.figure()
            frames = []

        # run episodes
        rewards_sum = []
        rewards_mean = []
        rewards_std = []
        for i in range(num_episodes):

            # reset environment
            state = self.env.reset()[0] # tuple contains as first element the state
            rewards = []
            j = 0
            while True:
                # take action and update environment
                action = self.agent.computeAction(state)
                next_state, reward, term, trunc, info = self.env.step(action=action)

                # learn q-network
                if self.learn_qval:
                    # add transition to replay buffer
                    self.buffer.addTransition(state=state, action=action, reward=reward, next_state=next_state, trunc=trunc)
                    batch = self.buffer.sampleBatch(batch_size=self.batch_size)

                    # train q-network if replay buffer is large enough
                    if batch is not False:
                        self.q_values.trainStep(batch=batch, agent=self.agent)
                
                # render environment
                if self.render:
                    env_screen = self.env.render()
                    frames.append([plt.imshow(env_screen)])

                # print info
                if self.verb:
                    print(f"step: {j}, action: {action}, reward: {reward}, terminated: {term}, truncated: {trunc}, info: {info}")

                # update state
                state = next_state
                rewards.append(reward)
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

            # add mean and std of rewards
            rewards_mean.append(np.mean(rewards))
            rewards_sum.append(np.sum(rewards))
            rewards_std.append(np.std(rewards))

        # show animation of environment
        if self.render:
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
            plt.show()

        # plot rewards
        if self.plot:
            fig = plt.figure()
            plt.errorbar(x=range(num_episodes), y=rewards_mean, yerr=rewards_std, ecolor="red", label="Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.legend()
            plt.show()

        # print statistics
        if self.stat:
            print(f"Statistics for {num_episodes} episodes")
            print(f"    Mean (of all episodes) cumulative reward: {np.mean(rewards_sum)}")
            print(f"    Std (of all episodes) cumulative reward: {np.std(rewards_sum)}")

        # plot q-values learning loss
        if self.learn_qval:
            self.q_values.plotLoss()

        return rewards_sum
    
