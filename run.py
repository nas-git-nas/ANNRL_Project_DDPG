import gym
import numpy as np
import matplotlib.pyplot as plt
import os

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
    buffer = ReplayBuffer(buffer_size=100000, seed=1)
    actor = RandomActor()

    # run algorithm
    simu = Simulation(
        dir_path="results/3_1_random", 
        env=env, 
        critic = critic,
        actor = actor, 
        buffer=buffer,
    )
    step_rewards, cum_rewards = simu.run(num_episodes=10, render=False, plot=True)

    print(f"Mean cummulative reward: {np.mean(cum_rewards)}, std: {np.std(cum_rewards)}")

def heuristic_pendulum_actor():
    torques = np.linspace(0, 1, 11)

    # create environment and actor
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = None
    buffer = ReplayBuffer(buffer_size=100000, seed=1)

    cum_sums = []
    cum_stds = []
    step_sums = []
    step_stds = []
    for torque in torques:
        # create actor
        actor = HeuristicActor(const_torque=torque)

        # run algorithm
        simu = Simulation(
            dir_path="results/3_2_heuristic", 
            env=env, 
            critic = critic,
            actor = actor, 
            buffer=buffer,
        )
        step_rewards, cum_rewards = simu.run(num_episodes=10, render=False, plot=False)

        # save mean and std of cummulative and step rewards
        cum_sums.append(np.mean(cum_rewards))
        cum_stds.append(np.std(cum_rewards))
        step_sums.append(np.mean(step_rewards))
        step_stds.append(np.std(step_rewards))

    # transform torques in action space [0, 1] to torque space [0, 2]
    torques = torques * 2

    # plot results
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    axs[0].errorbar(x=torques, y=cum_sums, yerr=cum_stds, ecolor="red", label="Mean and std")
    axs[0].set_xlabel("Constant Torque")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].set_title("Cummulative reward")

    axs[1].errorbar(x=torques, y=step_sums, yerr=step_stds, ecolor="red", label="Mean and std")
    axs[1].set_xlabel("Constant Torque")
    axs[1].set_ylabel("Reward")
    axs[1].legend()
    axs[1].set_title("Average reward")
    
    plt.savefig(os.path.join(simu.dir_path, "reward.pdf"))

def heuristic_qvalues_actor():
    # create environment, critic, actor, noise and buffer
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=1.0)
    actor = HeuristicActor(const_torque=1.0)
    buffer = ReplayBuffer(buffer_size=10000, seed=1)

    # train algorithm
    simu = Simulation(
        dir_path="results/4_qvalues", 
        env=env, 
        critic = critic,
        actor = actor, 
        buffer=buffer,
    )
    simu.train(num_episodes=10, batch_size=128)

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
        dir_path="results/6_target_ddpg", 
        env=env, 
        critic = critic,
        actor = actor, 
        buffer=buffer,
    )
    simu.train(num_episodes=1000, batch_size=128)

def ou_ddpg():
    # create environment, critic, actor, noise and buffer
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=1.0)
    action_noise = OUActionNoise(sigma=0.3, theta=0.5, seed=0)
    actor = Actor(lr=1e-4, tau=0.1, noise=action_noise)
    buffer = ReplayBuffer(buffer_size=100000, seed=1)

    # train algorithm
    simu = Simulation(
        dir_path="results/ou_noise_ddpg", 
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
    heuristic_qvalues_actor()

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
        Orstein-Uhlenbeck noise
        -> test and debug
    """
    # ou_ddpg()