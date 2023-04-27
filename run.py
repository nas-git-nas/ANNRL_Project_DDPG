import gym
import numpy as np
import matplotlib.pyplot as plt

from src.environment import NormalizedEnv
from src.action import RandomAgent, HeuristicPendulumAgent, DDPGAgent
from src.q_values import QValues
from src.simulation import Simulation
    

def random_agent():
    # create environment and agent
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    agent = RandomAgent(env=env)

    # run algorithm
    simu = Simulation(env=env, agent=agent, verb=False, render=True, plot=True, stat=True)
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
        step_rewards = simu.run(num_episodes=10)

        # save mean and std of rewards
        sums.append(np.mean(step_rewards))
        stds.append(np.std(step_rewards))

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
    q_values = QValues(gamma=0.99, lr=1e-4)
    agent = HeuristicPendulumAgent(env=env, const_torque=0.2)

    # run algorithm
    simu = Simulation(env=env, agent=agent, q_values=q_values, verb=False, render=False, plot=False, stat=False)
    simu.run(num_episodes=60)

def simple_ddpg():
    # create environment and agent
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    q_values = QValues(gamma=0.99, lr=1e-4)
    agent = DDPGAgent(env=env, q_values=q_values, lr=1e-4)

    # run algorithm
    simu = Simulation(env=env, agent=agent, q_values=q_values, verb=False, render=False, plot=True, stat=False)
    simu.train(num_episodes=500)

def target_ddpg():
    # create environment and agent
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    q_values = QValues(gamma=0.99, lr=1e-4, tau=1)
    agent = DDPGAgent(env=env, q_values=q_values, lr=1e-4, tau=1)

    # run algorithm
    simu = Simulation(env=env, agent=agent, q_values=q_values, verb=False, render=False, plot=True, stat=False)
    simu.train(num_episodes=100)


if __name__ == "__main__":
    """
    PART 3
        Random input
    """
    # random_agent()

    """
    PART 3
        Heuristic input, grid search for best constant torque
    """
    # heuristic_pendulum_agent()

    """
    PART 4
        Q-values learning
        -> check loss curve (second peak)
        -> implement polar heatmaps
    """
    # heuristic_qvalues_agent()

    """
    PART 5
        Simple DDPG
        
    """
    # simple_ddpg()

    """
    PART 6
        Target DDPG

    """
    target_ddpg()