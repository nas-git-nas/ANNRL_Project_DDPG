import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from src.critic_network import CriticNetwork


class Critic():
    def __init__(self, gamma=0.99, lr=0.0001, tau=1.0):
        # hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
       
        # model
        self.qnet = CriticNetwork()
        self.loss_fct = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

        # target model
        assert 0.0 <= self.tau and self.tau <= 1.0, "tau must be between 0 and 1"
        if self.tau < 1.0:
            self.target_qnet = CriticNetwork()

        # logging values
        self.log_losses = []
        self.log_targets = []
        self.log_q_values = []

    def computeQValue(self, states, actions, use_target_network):
        # calculate expected cumulative rewards
        qnet_input = torch.cat((states, actions.reshape(-1,1)), dim=1)

        if use_target_network and self.tau < 1.0:
            return self.target_qnet.forward(qnet_input)
        else:
            return self.qnet.forward(qnet_input)  

    def trainStep(self, batch, actor):
        # do not train if relay buffer is not large enough
        if batch is None:
            self.log_losses.append(0)
            return      

        # calculate next actions and target, with torch.no_grad()
        targets = self._computeTarget(actor=actor, batch=batch)

        # calculate target and expected cumulative rewards
        q_values = self.computeQValue(states=batch["state"], actions=batch["action"], use_target_network=False)
        
        # calculate loss and log it
        loss = self.loss_fct(q_values, targets)
        self.log_losses.append(loss.item())
        # self.log_q_values = self.log_q_values + list(q_values.detach().numpy().flatten())
        # self.log_targets = self.log_targets + list(targets.detach().numpy().flatten())

        # backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self._updateTargetNetwork()
    
    def plotLoss(self, path):
        i = 0
        losses = []
        while i < len(self.log_losses):
            losses.append(np.mean(self.log_losses[i:i+200]))
            i += 200

        fig = plt.figure()
        plt.plot(losses, label="Avg. MSE Loss per episode", color="green")
        plt.title("Critic Loss")
        plt.xlabel("Episode")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig(os.path.join(path, "loss_critic.pdf"))

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
            v_val = self.computeQValue(states=state_copy, actions=torque, use_target_network=False)
            v_val = v_val.reshape(angle.shape).detach().numpy()
            
            colorbar = axs[i,j].pcolormesh(angle, vel, v_val)
            axs[i,j].axis([np.min(angle), np.max(angle), np.min(vel), np.max(vel)])
            fig.colorbar(colorbar, ax=axs[i,j])
            axs[i,j].set_xlabel("Angle [rad]")
            axs[i,j].set_ylabel("Velocity [rad/s]")
            axs[i,j].set_title(f"Torque = {2*t}Nm")

        plt.savefig(os.path.join(path, "heatmap.pdf"))

    def _computeTarget(self, actor, batch):

        with torch.no_grad():
            # calculate next actions
            next_actions = actor.computeAction(batch["next_state"], use_target_network=True, deterministic=True)

            # calculate next Q value, equal to 0 if truncated
            next_q_values = self.computeQValue(states=batch["next_state"], actions=next_actions, use_target_network=True)
            next_q_values = torch.where(batch["trunc"].reshape_as(next_q_values), 0, next_q_values)
            
            # calculate target
            targets = batch["reward"].reshape_as(next_q_values) + self.gamma * next_q_values
        
        return targets
    
    def _updateTargetNetwork(self):
        if self.tau == 1.0:
            return
        
        # get state dicts of both networks
        dict = self.qnet.state_dict()
        target_dict = self.target_qnet.state_dict()

        # calculate moving average of parameters
        for key in target_dict:
            target_dict[key] = self.tau * dict[key] + (1-self.tau) * target_dict[key]

        # update target network
        self.target_qnet.load_state_dict(target_dict)