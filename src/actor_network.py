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
    
