import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed

from src.environment import NormalizedEnv
from src.critic import Critic
from src.actor import Actor, RandomActor, HeuristicActor
from src.action_noise import GaussianActionNoise, OUActionNoise
from src.replay_buffer import ReplayBuffer
from src.simulation import Simulation

def plotLosses(dir, path, plot_actor=True):
    """
    Plot the losses and cumulative reward of a single model.

    Parameters:
        dir (str): path to the directory containing the losses.csv file
        path (str): path to the directory where the plot will be saved
        plot_actor (bool): whether to plot the actor loss or not; without actor cumulative reward is also not plotted
    """
    if plot_actor:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        df = pd.read_csv(os.path.join(dir, "losses.csv"))

        # calculate moving average of cumulative rewards across 10 episodes
        cum_rewards = df["cum_rewards"].rolling(30).mean()

        axs[0].plot(range(len(df["critic"])), df["critic"].to_numpy(), color="green", label="Critic loss")
        axs[0].plot(range(len(df["actor"])), df["actor"].to_numpy(), color="blue", label="Actor loss")
        axs[1].plot(range(len(df["cum_rewards"])), cum_rewards, color="blue", label="Cumulative reward")

        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].set_title(f"Losses")

        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Cumulative reward")
        axs[1].legend()
        axs[1].set_title(f"Cumulative reward")

        plt.savefig(os.path.join(path, "losses_rewards.pdf"), bbox_inches='tight')
    
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        df = pd.read_csv(os.path.join(dir, "losses.csv"))

        ax.plot(range(len(df["critic"])), df["critic"].to_numpy(), color="green", label="Critic loss")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Critic loss")
        
        plt.savefig(os.path.join(path, "critic_loss.pdf"), bbox_inches='tight')

def plotReplicateLosses(dirs, path):
    """
    Plot the losses and cumulative rewards of multiple runs of a single model.

    Parameters:
        dirs (list): list of paths to the directories containing the losses.csv files
        path (str): path to the directory where the plot will be saved
    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    for i, dir_path in enumerate(dirs):
        df = pd.read_csv(os.path.join(dir_path, "losses.csv"))

        # calculate moving average of cumulative rewards across 10 episodes
        cum_rewards = df["cum_rewards"].rolling(30).mean()

        axs[0].plot(range(len(df["critic"])), df["critic"].to_numpy(), linewidth=0.8, label=f"Run {i}")
        axs[1].plot(range(len(df["actor"])), df["actor"].to_numpy(), label=f"Run {i}")
        axs[2].plot(range(len(df["cum_rewards"])), cum_rewards, label=f"Run {i}")

    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Critic loss")
    axs[0].legend()
    axs[0].set_title(f"Critic loss")

    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Actor loss")
    axs[1].legend()
    axs[1].set_title(f"Actor loss")

    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Cumulative reward")
    axs[2].legend()
    axs[2].set_title(f"Cumulative reward")

    fig.tight_layout()

    plt.savefig(os.path.join(path, "replicate_losses.pdf"), bbox_inches='tight')

def plotMultipleLosses(dirs, path, labeltheta=True):
    """
    Plot the losses and cumulative rewards of multiple models.

    Parameters:
        dirs (list): list of paths to the directories containing the losses.csv files
        path (str): path to the directory where the plot will be saved
        labeltheta (bool): whether to label thetas or not
    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    for dir_path in dirs:
        df = pd.read_csv(os.path.join(dir_path, "losses.csv"))

        # calculate moving average of cumulative rewards across 10 episodes
        cum_rewards = df["cum_rewards"].rolling(30).mean()

        if labeltheta:
            axs[0].plot(range(len(df["critic"])), df["critic"].to_numpy(), linewidth=0.8, label="theta="+str(df["theta"][0]))
            axs[1].plot(range(len(df["actor"])), df["actor"].to_numpy(), label="theta="+str(df["theta"][0]))
            axs[2].plot(range(len(df["cum_rewards"])), cum_rewards, label="theta="+str(df["theta"][0]))
        else:
            axs[0].plot(range(len(df["critic"])), df["critic"].to_numpy(), linewidth=0.8, label="tau="+str(df["tau"][0]))
            axs[1].plot(range(len(df["actor"])), df["actor"].to_numpy(), label="tau="+str(df["tau"][0]))
            axs[2].plot(range(len(df["cum_rewards"])), cum_rewards, label="tau="+str(df["tau"][0]))

    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title(f"Critic loss")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].set_title(f"Actor loss")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Cumulative reward")
    axs[2].legend(loc="lower right")
    axs[2].set_title(f"Cumulative reward")

    fig.tight_layout()

    plt.savefig(os.path.join(path, "multiple_losses.pdf"), bbox_inches='tight')

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
    step_rewards, cum_rewards = simu.run(num_episodes=1000, render=False, plot=True)

    print(f"Mean cumulative reward: {np.mean(cum_rewards)}, std: {np.std(cum_rewards)}")

def heuristic_pendulum_actor(plot_random=True):
    if plot_random:
        torques = np.linspace(0, 1, 100)

        # create environment and actor
        env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
        critic = None
        buffer = ReplayBuffer(buffer_size=100000, seed=1)

        # run episodes with different torques in parallel
        run_experiment = lambda torque: np.array(Simulation(dir_path="results/3_2_heuristic",
                                            env=env,
                                            critic=critic,
                                            actor=HeuristicActor(const_torque=torque),
                                            buffer=buffer, create_dir=False).run(num_episodes=1000, render=False, plot=False, verbose=False)[1])
        cum_rewards = np.array(Parallel(n_jobs=10)(delayed(run_experiment)(torque) for torque in torques))
        cum_rewards_mean = cum_rewards.mean(axis=1)
        cum_rewards_std = cum_rewards.std(axis=1)

        # run episodes with random actor
        actor = RandomActor()
        simu = Simulation( 
            dir_path="results/3_1_random",
            env=env, 
            critic = critic,
            actor = actor, 
            buffer=buffer,
        )
        _, cum_rewards_random = simu.run(num_episodes=1000, render=False, plot=False, verbose=False)

        # print cumulative reward
        print(f"Mean cumulative reward for random agent: {np.array(cum_rewards_random).mean():.4f}, std: {np.array(cum_rewards_random).std():.4f}")
        print(f"Mean cumulative reward for |torque|=1: {cum_rewards_mean[-1]:.4f}, std: {cum_rewards_std[-1]:.4f}")

        # plot results
        fig, ax = plt.subplots(figsize=(5,5))
        ax.errorbar(x=torques*2, y=cum_rewards_mean, yerr=cum_rewards_std, ecolor="lightblue", label="heuristic agent, mean and std")
        ax.axhline(y = np.array(cum_rewards_random).mean(), color = 'r', linestyle = '-', label="random agent, mean")
        ax.legend(loc="lower right")
        ax.set_xlabel("Torque")
        ax.set_ylabel("Average cumulative reward in 1000 episodes")
        plt.savefig(os.path.join("results/3_2_heuristic", "heuristic_vs_random.pdf"), bbox_inches='tight')
    
    else:
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
            step_rewards, cum_rewards = simu.run(num_episodes=100, render=False, plot=False)

            # save mean and std of cumulative and step rewards
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
        axs[0].set_title("Cumulative reward")

        axs[1].errorbar(x=torques, y=step_sums, yerr=step_stds, ecolor="red", label="Mean and std")
        axs[1].set_xlabel("Constant Torque")
        axs[1].set_ylabel("Reward")
        axs[1].legend()
        axs[1].set_title("Average reward")
        
        plt.savefig(os.path.join(simu.dir_path, "reward.pdf"), bbox_inches='tight')

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
    simu.run(num_episodes=100, render=False, plot=True)

def target_ddpg():
    # target network update
    tau = 0.05

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

def ou_ddpg():
    # OU noise parameters
    theta = 1.
    # create environment, critic, actor, noise and buffer
    env = NormalizedEnv(env=gym.make("Pendulum-v1", render_mode="rgb_array"))
    critic = Critic(gamma=0.99, lr=1e-4, tau=0.01)
    action_noise = OUActionNoise(sigma=0.3, theta=theta, seed=0)
    actor = Actor(lr=1e-4, tau=0.01, noise=action_noise)
    buffer = ReplayBuffer(buffer_size=100000, seed=1)
    # train algorithm
    simu = Simulation(
        dir_path="results/7_ou_noise_ddpg", 
        env=env, 
        critic = critic,
        actor = actor, 
        buffer=buffer,
    )
    simu.train(num_episodes=1000, batch_size=128)
    simu.run(num_episodes=100, render=False, plot=True)

def plot_report():
    # plot Q-networks losses
    dir_qnet = "results/4_qvalues/20230527_1629"
    plotLosses(dir=dir_qnet, path=dir_qnet, plot_actor=False)

    # plot minimal DDPG losses for two runs
    dirs_ddpg = ["results/5_simple_ddpg/20230526_1531_tau1.0_theta1.0",
                 "results/5_simple_ddpg/20230526_1559_tau1.0_theta1.0"]
    plotReplicateLosses(dirs=dirs_ddpg, path="results/5_simple_ddpg")

    # plot target DDPG losses
    dirs_target = ["results/6_target_ddpg/20230526_1841_tau0.01_theta1.0",
                   "results/6_target_ddpg/20230527_1614_tau0.05_theta1.0",
                   "results/6_target_ddpg/20230526_1714_tau0.1_theta1.0",
                   "results/6_target_ddpg/20230526_1659_tau0.5_theta1.0",
                   "results/6_target_ddpg/20230526_1633_tau1.0_theta1.0"]
    plotMultipleLosses(dirs=dirs_target, path="results/6_target_ddpg", labeltheta=False)

    # plot OU noise DDPG losses
    dirs_ou = ["results/7_ou_noise_ddpg/20230527_1434_tau0.01_theta0.0",
               "results/7_ou_noise_ddpg/20230527_1407_tau0.01_theta0.25",
               "results/7_ou_noise_ddpg/20230527_1353_tau0.01_theta0.5",
               "results/7_ou_noise_ddpg/20230527_1334_tau0.01_theta0.75",
               "results/7_ou_noise_ddpg/20230527_1455_tau0.01_theta1.0"]
    plotMultipleLosses(dirs=dirs_ou, path="results/7_ou_noise_ddpg")       


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
    """
    # heuristic_qvalues_actor()

    """
    PART 5
        Simple DDPG
    """
    simple_ddpg()

    """
    PART 6
        Target DDPG
    """
    # target_ddpg()

    """
    PART 7
        Orstein-Uhlenbeck noise
    """
    # ou_ddpg()

    """
    FINAl PART
        Make figures for report
    """
    # plot_report()