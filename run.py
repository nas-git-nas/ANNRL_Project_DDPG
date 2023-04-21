import gym
import numpy as np
import matplotlib.pyplot as plt

from environment import NormalizedEnv
from action import RandomAgent, HeuristicPendulumAgent
from simulation import Simulation
    

def random_agent():
    # create environment and agent
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    agent = RandomAgent(env=env)

    # run algorithm
    simu = Simulation(env=env, agent=agent, verb=False, render=False, plot=True, stat=True)
    simu.run(num_episodes=10)

def heuristic_pendulum_agent():
    torques = np.linspace(0, 1, 11)

    # create environment and agent
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))

    sums = []
    stds = []
    for torque in torques:
        # create agent
        agent = HeuristicPendulumAgent(env=env, const_torque=torque)

        # run algorithm
        simu = Simulation(env=env, agent=agent, verb=False, render=False, plot=False, stat=True)
        rewards_sum = simu.run(num_episodes=10)

        # save mean and std of rewards
        sums.append(np.mean(rewards_sum))
        stds.append(np.std(rewards_sum))

    # transform torques in action space [0, 1] to torque space [0, 2]
    torques = torques * 2

    # plot results
    fig = plt.figure()
    plt.errorbar(x=torques, y=sums, yerr=stds, ecolor="red", label="Cumulative Reward")
    plt.xlabel("Constant Torque")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

def heuristic_qvalues_agent():
    # create environment and agent
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    agent = HeuristicPendulumAgent(env=env, const_torque=0.5)

    # run algorithm
    simu = Simulation(env=env, agent=agent, learn_qval=True, verb=False, render=False, plot=False, stat=True)
    simu.run(num_episodes=100)


if __name__ == "__main__":
    random_agent()
    # heuristic_pendulum_agent()
    # heuristic_qvalues_agent()