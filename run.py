import gym
import numpy as np
import matplotlib.pyplot as plt
import os

from src.environment import NormalizedEnv
from src.actor import RandomActor, HeuristicActor, DDPGActor
from src.critic import Critic
from src.gaussian_action_noise import GaussianActionNoise, OUActionNoise
from src.simulation import Simulation
    

def random_actor():
    # create environment and actor
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    actor = RandomActor(env=env)

    # run algorithm
    simu = Simulation(buffer_size=10000, dir_path="results/3_1_random",env=env, actor=actor, verb=False, render=True, plot=True, stat=True)
    simu.run(num_episodes=10)

def heuristic_pendulum_actor():
    torques = np.linspace(0, 1, 11)

    # create environment and actor
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))

    sums = []
    stds = []
    for torque in torques:
        # create actor
        actor = HeuristicActor(env=env, const_torque=torque)

        # run algorithm
        simu = Simulation(buffer_size=10000, dir_path="results/3_2_heuristic", env=env, actor=actor, verb=False, render=False, plot=False, stat=True)
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

def heuristic_qvalues_actor():
    # create environment and actor
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4)
    actor = HeuristicActor(env=env, const_torque=1.0)

    # run algorithm
    simu = Simulation(buffer_size=10000, dir_path="results/4_qvalues", env=env, actor=actor, critic=critic, verb=False, render=False, plot=True, stat=False)
    simu.train(num_episodes=1050)

def simple_ddpg():
    # create environment and actor
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=1.0)
    action_noise = GaussianActionNoise(sigma=0.3, seed=0)
    actor = DDPGActor(env=env, critic=critic, action_noise=action_noise, lr=1e-4, noise_std=0.3, tau=1.0)

    # train algorithm
    simu = Simulation(buffer_size=100000, dir_path="results/5_simple_ddpg", env=env, actor=actor, critic=critic, verb=False, render=False, plot=True, stat=False)
    simu.train(num_episodes=1500)

def target_ddpg():
    # create environment and actor
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=0.1)
    action_noise = GaussianActionNoise(sigma=0.3, seed=0)
    actor = DDPGActor(env=env, critic=critic, action_noise=action_noise, lr=1e-4, tau=0.1, noise_std=0.3)

    # train algorithm
    simu = Simulation(buffer_size=100000, dir_path="results/6_target_ddpg", env=env, actor=actor, critic=critic, verb=False, render=False, plot=True, stat=False)
    simu.train(num_episodes=1500)

def ou_noise_ddpg():
    # create environment and actor
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=0.1)
    action_noise = OUActionNoise(sigma=0.3, theta=0.15, seed=0)
    actor = DDPGActor(env=env, critic=critic, action_noise=action_noise, lr=1e-4, tau=0.1)

    # train algorithm
    simu = Simulation(buffer_size=10000, dir_path="results/7_ou_noise_ddpg", env=env, actor=actor, critic=critic, verb=False, render=False, plot=True, stat=False)
    simu.train(num_episodes=250)


if __name__ == "__main__":
    """
    PART 3
        Random input
    """
    # random_actor()

    """
    PART 3
        Heuristic input, grid search for best constant torque
    """
    # heuristic_pendulum_actor()

    """
    PART 4
        Q-values learning
        -> implement polar heatmaps
    """
    # heuristic_qvalues_actor()

    """
    PART 5
        Simple DDPG
        -> test and debug
    """
    # simple_ddpg()

    """
    PART 6
        Target DDPG
        -> test and debug
    """
    # target_ddpg()

    """
    PART 7
        Ornstein-Uhlenbeck noise
        -> test and debug
    """
    ou_noise_ddpg()