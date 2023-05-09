import torch
from abc import abstractmethod

class ActionNoise():
    def __init__(self, sigma, seed):
        self._sigma = sigma
        self.generator = torch.Generator().manual_seed(seed)

    @abstractmethod
    def getNoisyAction(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

class GaussianActionNoise(ActionNoise):
    def __init__(self, sigma, seed=0) -> None:
        super().__init__(sigma, seed)

    def getNoisyAction(self, action):
        noisy_action = action + torch.normal(torch.zeros_like(action), torch.ones_like(action)*self._sigma, generator=self.generator)
        return torch.clip(noisy_action, -1, 1)
    
    def reset(self):
        pass
    

class OUActionNoise(ActionNoise):
    def __init__(self, sigma, theta, seed=0):
        super().__init__(sigma, seed)

        self.theta = theta
        self.prev_noise = 0.0

    def getNoisyAction(self, action):
        noise  = (1-self.theta)*self.prev_noise + torch.normal(torch.zeros_like(action), torch.ones_like(action)*self._sigma, generator=self.generator)
        noisy_action = action + noise

        self.prev_noise = noise
        return torch.clip(noisy_action, -1, 1)
    
    def reset(self):
        self.prev_noise = 0.0