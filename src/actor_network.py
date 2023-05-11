import torch

class ActorNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = torch.nn.Linear(3, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, states):
        x = self.relu(self.fc1(states))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))
    
class RandomActor():
    def __init__(self):
        pass
        
    def forward(self, states):
        # generate random actions between -1 and 1
        actions = torch.rand((states.shape[0],1))
        return 2*actions - 1
    

class HeuristicActor():
    def __init__(self, const_torque=0.5):
        if const_torque > 1 or const_torque < 0:
            raise ValueError("Constant torque must be between 0 and 1")
        self.const_torque = const_torque

    def forward(self, states):
        # generate heuristic actions
        actions = torch.empty((states.shape[0], self.action_size))
        actions[:,0] = -torch.sign(states[:,0]) * torch.sign(states[:,2]) * self.const_torque
        return actions