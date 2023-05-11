import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from src.critic_network import CriticNetwork
from src.actor_network import ActorNetwork, RandomActor, HeuristicActor


class CriticActor():
    def __init__(self, gamma, lr, tau, critic_type, actor_type):
        # hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.critic_type = critic_type
        self.actor_type = actor_type
       
        # initialize critic networks
        if critic_type == "network":
            self.critic_net = CriticNetwork()
        elif critic_type == "None":
            self.critic_net = None
        else:
            raise ValueError("critic_type must be one of: network, None")
        
        # initialize actor networks
        if actor_type == "network":
            self.actor_net = ActorNetwork()
        elif actor_type == "random":
            self.actor_net = RandomActor()
        elif actor_type == "heuristic":
            self.actor_net = HeuristicActor()
        else:
            raise ValueError("actor_type must be one of: network, random, heuristic")

        # intialize critic and actor optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr)

        # initialize target networks
        self.critic_target_net = CriticNetwork()
        self.actor_target_net = ActorNetwork()

        # logging values
        self.critic_losses = []
        self.actor_losses = []


    def trainStep(self, batch):
        # do not train if relay buffer is not large enough
        if batch is None:
            self.critic_losses.append(0)
            self.actor_losses.append(0)
            return
                  
        if self.critic_type == "network":
            # gradient descent step for critic network
            self.critic_optimizer.zero_grad()
            critic_loss = self._computeCriticsLoss(batch=batch)
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_losses.append(critic_loss.item())
        else:
            self.critic_losses.append(0)

        if self.actor_type == "network":
            # freeze critic network to avoid unnecessary computations of gradients
            for p in self.critic_net.parameters():
                p.requires_grad = False        

            # gradient descent step for actor network
            self.actor_optimizer.zero_grad()
            actor_loss = self._computeActorLoss(batch=batch)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_losses.append(actor_loss.item())

            # unfreeze critic network
            for p in self.critic_net.parameters():
                p.requires_grad = True
        else:
            self.actor_losses.append(0)

        # update target network
        self._updateTargetNetworks()

    def _computeCriticsLoss(self, batch):
        # calculate next actions and target, with torch.no_grad()
        with torch.no_grad():
            # calculate next actions and next Q values
            if self.tau == 1.0:
                next_actions = self.actor_net(batch["next_state"])
                next_q_values = self.critic_net.forward(states=batch["next_state"], actions=next_actions)
            else:
                next_actions = self.actor_target_net(batch["next_state"])
                next_q_values = self.critic_target_net.forward(states=batch["next_state"], actions=next_actions)

            # set next Q values to 0 if episode is truncated              
            next_q_values = torch.where(batch["trunc"].reshape_as(next_q_values), 0, next_q_values)
            
            # calculate target
            targets = batch["reward"].reshape_as(next_q_values) + self.gamma * next_q_values

        # calculate target and expected cumulative rewards
        q_values = self.critic_net.forward(states=batch["state"], actions=batch["action"])
        
        # calculate loss and log it
        return 0.5 * torch.pow(q_values - targets, 2).mean()
    
    def _computeActorLoss(self, batch):
        # estimate action from state
        actions = self.actor_net.forward(states=batch['state'])

        # calculate q values for state-action pairs
        q_values = self.critic_net.forward(states=batch['state'], actions=actions)

        # calculate loss      
        return - q_values.mean()
    
    def _updateTargetNetworks(self):
        if self.tau == 1.0:
            return
        
        with torch.no_grad():
            # update critic target network
            if self.critic_type == "network":          
                critic_net_dict = self.critic_net.state_dict()
                critic_target_net_dict = self.critic_target_net.state_dict()
                for key in critic_target_net_dict:
                    critic_target_net_dict[key] = self.tau * critic_net_dict[key] + (1-self.tau) * critic_target_net_dict[key]
                self.critic_target_net.load_state_dict(critic_target_net_dict)

            # update actor target network
            if self.actor_type == "network":
                actor_net_dict = self.actor_net.state_dict()
                actor_target_net_dict = self.actor_target_net.state_dict()
                for key in actor_target_net_dict:
                    actor_target_net_dict[key] = self.tau * actor_net_dict[key] + (1-self.tau) * actor_target_net_dict[key]
                self.actor_target_net.load_state_dict(actor_target_net_dict)

    def saveModels(self, path):
        if hasattr(self, "critic_net"):
            torch.save(self.critic_net, os.path.join(path, "critic_net.pt"))
        if hasattr(self, "critic_target_net"):
            torch.save(self.critic_target_net, os.path.join(path, "critic_target_net.pt"))
        if hasattr(self, "actor_net"):
            torch.save(self.actor_net, os.path.join(path, "actor_net.pt"))
        if hasattr(self, "actor_target_net"):
            torch.save(self.actor_target_net, os.path.join(path, "actor_target_net.pt"))
    
    def plotLosses(self, path):
        i = 0
        c_losses = []
        a_losses = []
        while i < len(self.critic_losses):
            c_losses.append(np.mean(self.critic_losses[i:i+200]))
            a_losses.append(np.mean(self.actor_losses[i:i+200]))
            i += 200

        fig = plt.figure()
        plt.plot(c_losses, label="Critic Loss", color="green")
        plt.plot(a_losses, label="Actor Loss", color="blue")
        plt.title("Losses")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(path, "losses.pdf"))

    def plotHeatmap(self, path):
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

        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
        fig.suptitle('Estimated V-values')
        for idx, t in enumerate(np.linspace(-1, 1, 9)):
            i = idx // 3
            j = idx % 3
            state_copy = state.detach().clone()

            torque = t * torch.ones(state_copy.shape[0])
            v_val = self.critic_net.forward(states=state_copy, actions=torque)
            v_val = v_val.reshape(angle.shape).detach().numpy()
            
            colorbar = axs[i,j].pcolormesh(angle, vel, v_val)
            axs[i,j].axis([np.min(angle), np.max(angle), np.min(vel), np.max(vel)])
            fig.colorbar(colorbar, ax=axs[i,j])
            axs[i,j].set_xlabel("Angle [rad]")
            axs[i,j].set_ylabel("Velocity [rad/s]")
            axs[i,j].set_title(f"Torque = {2*t}Nm")

        plt.savefig(os.path.join(path, "heatmap.pdf"))
    
