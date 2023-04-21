import gym
import numpy as np
from abc import abstractmethod


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
            raise ValueError("State size does not match environment state space size")
    
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