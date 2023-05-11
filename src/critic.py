import torch
import numpy as np
import os
import matplotlib.pyplot as plt


from src.critic_network import CriticNetwork
# import src.actor as actor



class Critic():
    def __init__(self, gamma, lr, tau):
        # hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
       
        # initialize critic network
        self.critic_net = CriticNetwork()
        self.actor = None

        # intialize critic
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)

        # initialize target network
        self.critic_target_net = CriticNetwork()

        # logging values
        self.log_losses = []

    def saveModels(self, path):
        torch.save(self.critic_net, os.path.join(path, "critic_net.pt"))
        torch.save(self.critic_target_net, os.path.join(path, "critic_target_net.pt"))

    def computeQValues(self, states, actions, target=False):
        if target:
            return self.critic_target_net.forward(states=states, actions=actions)
        else:
            return self.critic_net.forward(states=states, actions=actions)

    def trainStep(self, batch, actor):
        # do not train if relay buffer is not large enough
        if batch is None:
            self.log_losses.append(0)
            return                

        # gradient descent step for critic network
        self.critic_optimizer.zero_grad()
        critic_loss = self._computeCriticsLoss(batch=batch, actor=actor)
        critic_loss.backward()
        self.critic_optimizer.step()

        # log loss
        self.log_losses.append(critic_loss.item())

        # update target network
        self._updateTargetNetworks()

    def _computeCriticsLoss(self, batch, actor):
        # calculate next actions and target, with torch.no_grad()
        with torch.no_grad():
            # calculate next actions and next Q values
            if self.tau == 1.0:
                next_actions = actor.computeActions(states=batch["next_state"], target=False, deterministic=True)
                next_q_values = self.computeQValues(states=batch["next_state"], actions=next_actions, target=False)
            else:
                next_actions = actor.computeActions(batch["next_state"], target=True, deterministic=True)
                next_q_values = self.computeQValues(states=batch["next_state"], actions=next_actions, target=True)

            # set next Q values to 0 if episode is truncated              
            next_q_values = torch.where(batch["trunc"].reshape_as(next_q_values), 0, next_q_values)
            
            # calculate target
            targets = batch["reward"].reshape_as(next_q_values) + self.gamma * next_q_values

        # calculate target and expected cumulative rewards
        q_values = self.computeQValues(states=batch["state"], actions=batch["action"], target=False)
        
        # calculate loss and log it
        return 0.5 * torch.pow(q_values - targets, 2).mean()
    
    def _updateTargetNetworks(self):
        if self.tau == 1.0:
            return
        
        with torch.no_grad():
            # update critic target network         
            critic_net_dict = self.critic_net.state_dict()
            critic_target_net_dict = self.critic_target_net.state_dict()
            for key in critic_target_net_dict:
                critic_target_net_dict[key] = self.tau * critic_net_dict[key] + (1-self.tau) * critic_target_net_dict[key]
            self.critic_target_net.load_state_dict(critic_target_net_dict)


    
    
    
