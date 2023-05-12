import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from src.actor_network import ActorNetwork
from src.action_noise import ActionNoise
# import src.critic as critic

# try:
#     Critic
# except NameError:
#     from src.critic import Critic

class RandomActor():
    def __init__(self):
        pass
        
    def computeActions(self, states, target=False, deterministic=True):
        if target or not deterministic:
            raise ValueError("Random actor does not have a target network or noise")

        # generate random actions between -1 and 1
        actions = torch.rand((states.shape[0],1))
        return 2*actions - 1
    

class HeuristicActor():
    def __init__(self, const_torque):
        if const_torque > 1 or const_torque < 0:
            raise ValueError("Constant torque must be between 0 and 1")
        self.const_torque = const_torque

        self.log_losses = []
    
    def saveModels(self, path):
        pass

    def computeActions(self, states, target=False, deterministic=True):
        # if target or not deterministic:
        #     raise ValueError("Heuristic actor does not have a target network or noise")
        
        # generate heuristic actions
        actions = torch.empty((states.shape[0], 1))
        actions[:,0] = -torch.sign(states[:,0]) * torch.sign(states[:,2]) * self.const_torque
        return actions
    
    def trainStep(self, batch, critic):
        self.log_losses.append(0)


class Actor():
    def __init__(self, lr, tau, noise:ActionNoise):
        # hyperparameters
        self.lr = lr
        self.tau = tau

        self.noise = noise
        
        # initialize actor networks
        self.actor_net = ActorNetwork()

        # intialize critic and actor optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr)

        # initialize target networks
        self.actor_target_net = ActorNetwork()

        # logging values
        self.log_losses = []

    def saveModels(self, path):
        torch.save(self.actor_net, os.path.join(path, "actor_net.pt"))
        torch.save(self.actor_target_net, os.path.join(path, "actor_target_net.pt"))

    def computeActions(self, states, target=False, deterministic=True):
        if target:
            actions = self.actor_target_net.forward(states=states)
        else:
            actions = self.actor_net.forward(states=states)

        if not deterministic:
            actions += self.noise.getNoisyAction(actions=actions)
        return actions

    def trainStep(self, batch:dict, critic):
        # do not train if relay buffer is not large enough
        if batch is None:
            self.log_losses.append(0)
            return

        # freeze critic network to avoid unnecessary computations of gradients
        for p in critic.critic_net.parameters():
            p.requires_grad = False        

        # gradient descent step for actor network
        self.actor_optimizer.zero_grad()
        actor_loss = self._computeActorLoss(batch=batch, critic=critic)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.log_losses.append(actor_loss.item())

        # unfreeze critic network
        for p in critic.critic_net.parameters():
            p.requires_grad = True

        # update target network
        self._updateTargetNetworks()
    
    def _computeActorLoss(self, batch, critic):
        # estimate action from state
        actions = self.computeActions(states=batch['state'], target=False, deterministic=True)

        # calculate q values for state-action pairs
        q_values = critic.computeQValues(states=batch['state'], actions=actions, target=False)

        # calculate loss      
        return - q_values.mean()
    
    def _updateTargetNetworks(self):
        if self.tau == 1.0:
            return
        
        with torch.no_grad():
            # update actor target network
            actor_net_dict = self.actor_net.state_dict()
            actor_target_net_dict = self.actor_target_net.state_dict()
            for key in actor_target_net_dict:
                actor_target_net_dict[key] = self.tau * actor_net_dict[key] + (1-self.tau) * actor_target_net_dict[key]
            self.actor_target_net.load_state_dict(actor_target_net_dict)


    
    def plotLosses(self, path):
        i = 0
        c_losses = []
        a_losses = []
        while i < len(self.critic_losses):
            c_losses.append(np.mean(self.critic_losses[i:i+200]))
            a_losses.append(np.mean(self.log_losses[i:i+200]))
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
    
