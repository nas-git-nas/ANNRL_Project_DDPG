import numpy as np
import torch
from abc import abstractmethod
import matplotlib.pyplot as plt

from src.policy_network import PolicyNetwork
from src.q_values import QValues
from src.gaussian_action_noise import GaussianActionNoise
from src.environment import NormalizedEnv


class Agent():
    def __init__(self, env) -> None:
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.action_max = 1
        self.action_min = -1

    @abstractmethod
    def computeAction(self, state):
        pass

    def checkActions(self, actions):
        if actions.shape[1] != self.action_size:
            raise ValueError("Action size does not match environment action space size")
    
        if np.any(actions > self.action_max) or np.any(actions < self.action_min):
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

class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        
    def computeAction(self, state):
        # convert state to batch if necessary
        states = self.checkStates(state)

        actions = np.random.uniform(-1, 1, size=(states.shape[0],self.action_size))

        return self.checkActions(actions)
    

class HeuristicPendulumAgent(Agent):
    def __init__(self, env, const_torque=0.5):
        super().__init__(env)

        if const_torque > 1 or const_torque < 0:
            raise ValueError("Constant torque must be between 0 and 1")
        self.const_torque = const_torque

    def computeAction(self, state):
        # convert state to batch if necessary
        states = self.checkStates(state)

        actions = np.empty((states.shape[0], self.action_size))
        actions[:,0] = -np.sign(states[:,0]) * np.sign(states[:,2]) * self.const_torque

        return self.checkActions(actions)
    

class DDPGAgent(Agent):
    def __init__(self, env: NormalizedEnv, q_values: QValues, lr: float, tau:float=1) -> None:
        super().__init__(env)

        # hyperparameters
        self.lr = lr
        self.tau = tau

        # models
        self.policy_net = PolicyNetwork()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q_values = q_values
        self.gaussian_action_noise = GaussianActionNoise(sigma=0.3)

        # target models
        self.target_policy_net = PolicyNetwork()

        # logging values
        self.log_losses = []

    def computeAction(self, state, deterministic=True, convert_to_numpy=True):
        # convert state to batch if necessary
        states = self.checkStates(state)

        # estimate action from state
        actions = self.target_policy_net(torch.tensor(states, dtype=torch.float))

        # add noise if not deterministic
        if not deterministic:
            actions = self.gaussian_action_noise.getNoisyAction(actions)

        # convert to numpy array if necessary
        if convert_to_numpy:
            actions = actions.detach().numpy()
            actions = self.checkActions(actions)

        return actions
    
    def trainStep(self, batch):
        # do not train if relay buffer is not large enough
        if batch is False:
            self.log_losses.append(0)
            return
        
        # estimate action from state
        actions = self.policy_net(torch.tensor(batch['state'], dtype=torch.float))

        # calculate loss with q value network and log it
        exp_cum_rewards = self.q_values.computeQValue(states=batch['state'], actions=actions, use_target_network=False)
        loss = - torch.mean(exp_cum_rewards)
        self.log_losses.append(loss.item())

        # backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self._updateTargetNetwork()

    def plotLoss(self):
        # average losses of one episode
        i = 0
        losses = []
        while i < len(self.log_losses):
            losses.append(np.mean(self.log_losses[i:i+200]))
            i += 200

        fig = plt.figure()
        plt.plot(losses, label="Avg. MSE Loss per episode", color="green")
        plt.xlabel("Episode")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.show()

    def _updateTargetNetwork(self):
        # get state dicts of both networks
        dict = self.policy_net.state_dict()
        target_dict = self.target_policy_net.state_dict()

        # calculate moving average of parameters
        for key in target_dict:
            target_dict[key] = self.tau * dict[key] + (1-self.tau) * target_dict[key]

        # update target network
        self.target_policy_net.load_state_dict(target_dict)

    
        
