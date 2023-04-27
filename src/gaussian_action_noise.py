import torch

class GaussianActionNoise():
    def __init__(self, sigma, seed=0) -> None:
        self._sigma = sigma
        self.generator = torch.Generator().manual_seed(seed)


    def getNoisyAction(self, action):
        noisy_action = action + torch.normal(torch.zeros_like(action), torch.ones_like(action)*self._sigma, generator=self.generator)
        return torch.clip(noisy_action, -1, 1)