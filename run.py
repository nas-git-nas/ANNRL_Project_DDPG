import gym
import numpy as np
import matplotlib.pyplot as plt

from src.environment import NormalizedEnv
from src.critic import Critic
from src.actor import Actor, RandomActor, HeuristicActor
from src.action_noise import GaussianActionNoise, OUActionNoise
from src.replay_buffer import ReplayBuffer
from src.simulation import Simulation

def random_actor():
    # create environment and actor
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = None
    buffer = None
    actor = RandomActor()

    # run algorithm
    simu = Simulation(
        dir_path="results/5_simple_ddpg", 
        env=env, 
        critic = critic,
        actor = actor, 
        buffer=buffer,
    )
    step_rewards = simu.run(num_episodes=10, render=False)

def heuristic_pendulum_actor():
    torques = np.linspace(0, 1, 11)

    # create environment and actor
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = None
    buffer = None

    sums = []
    stds = []
    for torque in torques:
        # create actor
        actor = HeuristicActor(const_torque=torque)

        # run algorithm
        simu = Simulation(
            dir_path="results/5_simple_ddpg", 
            env=env, 
            critic = critic,
            actor = actor, 
            buffer=buffer,
        )
        step_rewards = simu.run(num_episodes=10, render=False)

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
    # create environment, critic, actor, noise and buffer
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=1.0)
    actor = HeuristicActor(const_torque=1.0)
    buffer = ReplayBuffer(buffer_size=100000, seed=1)

    # train algorithm
    simu = Simulation(
        dir_path="results/5_simple_ddpg", 
        env=env, 
        critic = critic,
        actor = actor, 
        buffer=buffer,
    )
    simu.train(num_episodes=1000, batch_size=128)

def simple_ddpg():
    # create environment, critic, actor, noise and buffer
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=1.0)
    action_noise = GaussianActionNoise(sigma=0.3, seed=0)
    actor = Actor(lr=1e-4, tau=1.0, noise=action_noise)
    buffer = ReplayBuffer(buffer_size=100000, seed=1)

    # train algorithm
    simu = Simulation(
        dir_path="results/5_simple_ddpg", 
        env=env, 
        critic = critic,
        actor = actor, 
        buffer=buffer,
    )
    simu.train(num_episodes=1000, batch_size=128)

def target_ddpg():
    # create environment, critic, actor, noise and buffer
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=1.0)
    action_noise = GaussianActionNoise(sigma=0.3, seed=0)
    actor = Actor(lr=1e-4, tau=0.1, noise=action_noise)
    buffer = ReplayBuffer(buffer_size=100000, seed=1)

    # train algorithm
    simu = Simulation(
        dir_path="results/5_simple_ddpg", 
        env=env, 
        critic = critic,
        actor = actor, 
        buffer=buffer,
    )
    simu.train(num_episodes=1000, batch_size=128)


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
    target_ddpg()