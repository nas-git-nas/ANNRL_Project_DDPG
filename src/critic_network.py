import torch


class CriticNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = torch.nn.Linear(4, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, states, actions):
        x = torch.cat((states, actions.reshape(-1,1)), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
