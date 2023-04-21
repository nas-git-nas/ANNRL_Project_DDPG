import torch
import numpy as np
import matplotlib.pyplot as plt

from q_network import QNetwork


class QValues():
    def __init__(self, gamma=0.99, lr=0.01):
        # hyperparameters
        self.gamma = gamma
        self.lr = lr
       
       # model
        self.qnet = QNetwork()
        self.loss_fct = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=self.lr)

        # logging values
        self.log_losses = []
        

    def trainStep(self, batch, agent):
        # calculate next actions
        next_action = agent.computeAction(batch["next_state"])

        # calculate target and expected cumulative rewards
        exp_cum_rewards = self.estimate(states=batch["state"], actions=batch["action"])
        targets = self._calcTarget(next_states=batch["next_state"], next_actions=next_action, 
                                   rewards=batch["reward"], truncs=batch["trunc"])
        
        # calculate loss and backpropagate it
        loss = self.loss_fct(exp_cum_rewards, targets)
        loss.backward()
        self.optimizer.step()
        self.qnet.zero_grad()

        # log loss
        self.log_losses.append(loss.item())

    def estimate(self, states, actions):
        qnet_input = torch.cat((torch.tensor(states, dtype=torch.float), 
                                torch.tensor(actions, dtype=torch.float).reshape(-1,1)), dim=1)
        return self.qnet.forward(qnet_input)
    
    def plotLoss(self):
        fig = plt.figure()
        plt.plot(self.log_losses)
        plt.xlabel("Step")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.show()

    def _calcTarget(self, next_states, next_actions, rewards, truncs):
        with torch.no_grad():
            # calculate expected cumulative reward, equal to 0 if truncated
            next_exp_cum_rewards = self.estimate(states=next_states, actions=next_actions)
            next_exp_cum_rewards = torch.where(torch.from_numpy(truncs), 0, next_exp_cum_rewards)
            
            # calculate target
            target = torch.tensor(rewards, dtype=torch.float) + self.gamma * next_exp_cum_rewards
        
        return target
    
    # def _batch2Array(self, batch):
    #     batch_size = batch.shape[0]

    #     states = np.vstack(batch[:,0]).reshape(batch_size, 3)
    #     actions = np.vstack(batch[:,1]).reshape(batch_size, 1)
    #     rewards = np.array(batch[:,2], dtype=float).reshape(batch_size, 1)
    #     next_states = np.vstack(batch[:,3]).reshape(batch_size, 3)
    #     truncs = np.array(batch[:,4], dtype=bool).reshape(batch_size, 1)
    #     return states, actions, rewards, next_states, truncs