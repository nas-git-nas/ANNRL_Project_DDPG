import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.environment import NormalizedEnv
from src.critic import Critic
from src.actor import Actor, RandomActor, HeuristicActor
from src.action_noise import GaussianActionNoise, OUActionNoise
from src.replay_buffer import ReplayBuffer
from src.simulation import Simulation

def plotAllLosses(dirs, path):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    fig.suptitle("Losses")
    for dir_path in dirs:
        df = pd.read_csv(os.path.join(dir_path, "losses.csv"))
        axs[0].plot(range(len(df["critic"])), df["critic"].to_numpy(), label="tau="+str(df["tau"][0]) + ", theta="+str(df["theta"][0]))
        axs[1].plot(range(len(df["actor"])), df["actor"].to_numpy(), label="tau="+str(df["tau"][0]) + ", theta="+str(df["theta"][0]))

    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title(f"Critic loss")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].set_title(f"Actor loss")

    plt.savefig(os.path.join(path, "losses.pdf"))

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
    simu.train(num_episodes=200, batch_size=128)

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
    simu.run(num_episodes=100, render=False, plot=True)

def target_ddpg():
    # target network update
    tau = 1.0

    # create environment, critic, actor, noise and buffer
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=tau)
    action_noise = GaussianActionNoise(sigma=0.3, seed=0)
    actor = Actor(lr=1e-4, tau=tau, noise=action_noise)
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
    simu.run(num_episodes=100, render=False, plot=True)

    # dirs = ["results/6_target_ddpg/20230525_0926_tau0.01_theta1.0", 
    #         "results/6_target_ddpg/20230525_0949_tau0.05_theta1.0",
    #         "results/6_target_ddpg/20230525_1013_tau0.1_theta1.0",
    #         "results/6_target_ddpg/20230525_1035_tau0.5_theta1.0",
    #         "results/6_target_ddpg/20230525_1055_tau1.0_theta1.0"]
    # plotAllLosses(dirs=dirs, path="results/6_target_ddpg")


def ou_ddpg():
    # # OU noise parameters
    # theta = 1.0

    # # create environment, critic, actor, noise and buffer
    # env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    # critic = Critic(gamma=0.99, lr=1e-4, tau=0.01)
    # action_noise = OUActionNoise(sigma=0.3, theta=theta, seed=0)
    # actor = Actor(lr=1e-4, tau=0.01, noise=action_noise)
    # buffer = ReplayBuffer(buffer_size=100000, seed=1)

    # # train algorithm
    # simu = Simulation(
    #     dir_path="results/7_ou_noise_ddpg", 
    #     env=env, 
    #     critic = critic,
    #     actor = actor, 
    #     buffer=buffer,
    # )
    # simu.train(num_episodes=1000, batch_size=128)
    # simu.run(num_episodes=100, render=False, plot=True)

    dirs = ["results/7_ou_noise_ddpg/20230525_1305_tau0.01_theta0.0",
            "results/7_ou_noise_ddpg/20230525_1325_tau0.01_theta0.25",
            "results/7_ou_noise_ddpg/20230525_1344_tau0.01_theta0.5",
            "results/7_ou_noise_ddpg/20230525_1401_tau0.01_theta0.75",
            "results/7_ou_noise_ddpg/20230525_1511_tau0.01_theta1.0"]
    plotAllLosses(dirs=dirs, path="results/7_ou_noise_ddpg")


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
        Orstein-Uhlenbeck noise
        -> test and debug
    """
    ou_ddpg()