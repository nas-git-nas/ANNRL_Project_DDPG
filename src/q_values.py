import torch
import numpy as np
import matplotlib.pyplot as plt

from src.q_network import QNetwork


class QValues():
    def __init__(self, gamma=0.99, lr=0.0001, tau=1):
        # hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
       
        # model
        self.qnet = QNetwork()
        self.loss_fct = torch.nn.MSELoss()
        # self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

        # target model
        self.target_qnet = QNetwork()


        # logging values
        self.log_losses = []
        self.log_targets = []
        self.log_exp_cum_rewards = []
        

    def trainStep(self, batch, agent):
        # do not train if relay buffer is not large enough
        if batch is False:
            self.log_losses.append(0)
            return

        # calculate next actions
        next_action = agent.computeAction(batch["next_state"])

        # calculate target and expected cumulative rewards
        exp_cum_rewards = self.computeQValue(states=batch["state"], actions=batch["action"])
        targets = self._calcTarget(next_states=batch["next_state"], next_actions=next_action, 
                                   rewards=batch["reward"], truncs=batch["trunc"])
        
        # calculate loss and log it
        loss = self.loss_fct(exp_cum_rewards, targets)
        self.log_losses.append(loss.item())
        # self.log_exp_cum_rewards = self.log_exp_cum_rewards + list(exp_cum_rewards.detach().numpy().flatten())
        # self.log_targets = self.log_targets + list(targets.detach().numpy().flatten())

        # backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self._updateTargetNetwork()
        

    def computeQValue(self, states, actions, use_target_network=True):
        # convert states and actions to tensors if necessary
        if not torch.is_tensor(states):
            states = torch.tensor(states, dtype=torch.float)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.float)

        # calculate expected cumulative rewards
        qnet_input = torch.cat((states, actions.reshape(-1,1)), dim=1)

        if use_target_network:
            return self.target_qnet.forward(qnet_input)
        else:
            return self.qnet.forward(qnet_input)
    
    def plotLoss(self):
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

        # fig = plt.figure()
        # rewards_avg = np.convolve(self.log_exp_cum_rewards, np.ones(100)/100, mode='valid')
        # plt.plot(rewards_avg, label="avg exp cum reward", color="blue")
        # targets_avg = np.convolve(self.log_targets, np.ones(100)/100, mode='valid')
        # plt.plot(targets_avg, label="avg target", color="red")
        # plt.xlabel("Step")
        # plt.ylabel("reward")
        # plt.legend()
        # plt.show()

    def plotHeatmap(self):
        res = 20

        angle = np.linspace(-np.pi, np.pi, res)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        vel = np.linspace(-5, 5, res)

        meshgrid = np.meshgrid(np.arange(res), np.arange(res))
        idxs_a = meshgrid[0].flatten()
        idxs_vel = meshgrid[1].flatten()

        angle = angle[idxs_a]
        cos_a = cos_a[idxs_a]
        sin_a = sin_a[idxs_a]
        vel = vel[idxs_vel]

        state = np.concatenate((cos_a.reshape(-1,1), sin_a.reshape(-1,1), vel.reshape(-1,1)), axis=1)
        torque = np.ones(state.shape[0]) * 0
        v_val = self.computeQValue(states=state, actions=torque)

        angle = angle.reshape(meshgrid[0].shape)
        vel = vel.reshape(angle.shape)
        v_val = v_val.reshape(angle.shape).detach().numpy()

        fig, ax = plt.subplots()
        colorbar = ax.pcolormesh(angle, vel, v_val)
        ax.axis([np.min(angle), np.max(angle), np.min(vel), np.max(vel)])
        fig.colorbar(colorbar, ax=ax)
        ax.set_xlabel("Angle [rad]")
        ax.set_ylabel("Velocity [rad/s]")
        ax.set_title("Estimated V-values")
        plt.show()

    def _calcTarget(self, next_states, next_actions, rewards, truncs):
        with torch.no_grad():
            # calculate expected cumulative reward, equal to 0 if truncated
            next_exp_cum_rewards = self.computeQValue(states=next_states, actions=next_actions, use_target_network=False)
            next_exp_cum_rewards = torch.where(torch.from_numpy(truncs).reshape_as(next_exp_cum_rewards), 
                                               0, next_exp_cum_rewards)
            
            # calculate target
            target = torch.tensor(rewards, dtype=torch.float).reshape_as(next_exp_cum_rewards) \
                        + self.gamma * next_exp_cum_rewards
        
        return target
    
    def _updateTargetNetwork(self):
        # get state dicts of both networks
        dict = self.qnet.state_dict()
        target_dict = self.target_qnet.state_dict()

        # calculate moving average of parameters
        for key in target_dict:
            target_dict[key] = self.tau * dict[key] + (1-self.tau) * target_dict[key]

        # update target network
        self.target_qnet.load_state_dict(target_dict)
    


    
    # def _MSE(self, exp_cum_rewards, targets):
    #     loss = torch.pow((targets-exp_cum_rewards), 2)
    #     return torch.mean(loss)
    
    # def _batch2Array(self, batch):
    #     batch_size = batch.shape[0]

    #     states = np.vstack(batch[:,0]).reshape(batch_size, 3)
    #     actions = np.vstack(batch[:,1]).reshape(batch_size, 1)
    #     rewards = np.array(batch[:,2], dtype=float).reshape(batch_size, 1)
    #     next_states = np.vstack(batch[:,3]).reshape(batch_size, 3)
    #     truncs = np.array(batch[:,4], dtype=bool).reshape(batch_size, 1)
    #     return states, actions, rewards, next_states, truncs