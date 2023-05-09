import numpy as np
import torch
from abc import abstractmethod
import os
import matplotlib.pyplot as plt

from src.actor_network import ActorNetwork
from src.critic import Critic
from src.gaussian_action_noise import ActionNoise
from src.environment import NormalizedEnv


class Actor():
    def __init__(self, env:NormalizedEnv) -> None:
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.action_max = 1
        self.action_min = -1

        # logging values
        self.log_losses = []

    @abstractmethod
    def computeAction(self, state, use_target_network, deterministic):
        pass

    @abstractmethod
    def trainStep(self, batch):
        pass

    def checkActions(self, actions):
        if actions.shape[1] != self.action_size:
            raise ValueError("Action size does not match environment action space size")
    
        if torch.any(actions > self.action_max) or torch.any(actions < self.action_min):
            raise ValueError("Action is out of bounds")
        
        return actions.squeeze()
        
    def checkStates(self, states):
        if len(states.shape) == 2:
            if states.shape[1] != self.state_size:
                raise ValueError("State size does not match environment state space size")
            return states
        elif len(states.shape) == 1:
            if states.shape[0] != self.state_size:
                raise ValueError("State size does not match environment state space size")
            return states.reshape(1, 3)
        else:
            raise ValueError("State must be a 1D or 2D numpy array")
        
    def plotLoss(self, path):
        # average losses of one episode
        i = 0
        losses = []
        while i < len(self.log_losses):
            losses.append(np.mean(self.log_losses[i:i+200]))
            i += 200

        fig = plt.figure()
        plt.plot(losses, label="Avg. MSE Loss per episode", color="green")
        plt.title("Actor Loss")
        plt.xlabel("Episode")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig(os.path.join(path, "loss_actor.pdf"))

class RandomActor(Actor):
    def __init__(self, env:NormalizedEnv):
        super().__init__(env)
        
    def computeAction(self, state, use_target_network, deterministic):
        # convert state to batch if necessary
        states = self.checkStates(state)

        actions = np.random.uniform(-1, 1, size=(states.shape[0],self.action_size))

        return self.checkActions(actions)
    
    def trainStep(self, batch):
        # do not train
        self.log_losses.append(0)
    

class HeuristicActor(Actor):
    def __init__(self, env:NormalizedEnv, const_torque=0.5):
        super().__init__(env)

        if const_torque > 1 or const_torque < 0:
            raise ValueError("Constant torque must be between 0 and 1")
        self.const_torque = const_torque

    def computeAction(self, state, use_target_network, deterministic):
        # convert state to batch if necessary
        states = self.checkStates(state)

        actions = torch.empty((states.shape[0], self.action_size))
        actions[:,0] = -torch.sign(states[:,0]) * torch.sign(states[:,2]) * self.const_torque

        return self.checkActions(actions)
    
    def trainStep(self, batch):
        # do not train
        self.log_losses.append(0)
    

class DDPGActor(Actor):
    def __init__(self, env:NormalizedEnv, critic:Critic, action_noise:ActionNoise, lr:float, tau:float=1.0) -> None:
        super().__init__(env)

        # hyperparameters
        self.lr = lr
        self.tau = tau

        # models
        self.pnet = ActorNetwork()
        self.optimizer = torch.optim.Adam(self.pnet.parameters(), lr=self.lr)
        self.critic = critic
        self.action_noise = action_noise

        # target models
        assert 0.0 <= self.tau and self.tau <= 1.0, "tau must be between 0 and 1"
        if self.tau < 1.0:
            self.target_pnet = ActorNetwork() 

    def computeAction(self, state, use_target_network, deterministic):
        # # convert state to batch if necessary
        # states = self.checkStates(state)

        # estimate action from state
        if use_target_network and self.tau < 1.0:
            action = self.target_pnet(state)
        else:
            action = self.pnet(state)

        # add noise if not deterministic
        if not deterministic:
            action = self.action_noise.getNoisyAction(action)

        return action
    
    def trainStep(self, batch):
        # do not train if relay buffer is not large enough
        if batch is None:
            self.log_losses.append(0)
            return
        
        # estimate action from state
        actions = self.computeAction(batch['state'], deterministic=True, use_target_network=False)

        # calculate loss with q value network and log it
        q_values = self.critic.computeQValue(states=batch['state'], actions=actions, use_target_network=False)
        loss = - torch.mean(q_values)
        self.log_losses.append(loss.item())

        # backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self._updateTargetNetwork()

    def _updateTargetNetwork(self):
        if self.tau == 1.0:
            return
        
        # get state dicts of both networks
        dict = self.pnet.state_dict()
        target_dict = self.target_pnet.state_dict()

        # calculate moving average of parameters
        for key in target_dict:
            target_dict[key] = self.tau * dict[key] + (1-self.tau) * target_dict[key]

        # update target network
        self.target_pnet.load_state_dict(target_dict)

    
        
