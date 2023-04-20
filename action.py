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

    def checkAction(self, action):
        if action.shape[0] != self.action_size:
            raise ValueError("Action size does not match environment action space size")     
        if np.any(action > self.action_max) or np.any(action < self.action_min):
            raise ValueError("Action is out of bounds")
        
    def checkState(self, state):
        if state.shape[0] != self.state_size:
            raise ValueError("State size does not match environment state space size")
        if len(state.shape) > 1:
            raise ValueError("State must be a 1D numpy array")

class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        
    def computeAction(self, state):
        action = np.random.uniform(-1, 1, size=(self.action_size, ))
        self.checkAction(action)
        return action
    

class HeuristicPendulumAgent(Agent):
    def __init__(self, env, const_torque=0.5):
        super().__init__(env)

        if const_torque > 1 or const_torque < 0:
            raise ValueError("Constant torque must be between 0 and 1")
        self.const_torque = const_torque

    def computeAction(self, state):
        self.checkState(state)

        if state[0] > 0:
            action = - np.sign(state[2]) * self.const_torque
        else:
            action = np.sign(state[2]) * self.const_torque
        action = np.array([action]) # because action must be an array

        self.checkAction(action)
        return action