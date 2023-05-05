import numpy as np
import torch
import random

# Look into: collection deque


class ReplayBuffer():
    def __init__(self, buffer_size, seed):     
        self._buffer = { 
            "state" : torch.empty((buffer_size,3), dtype=torch.float32, requires_grad=False),
            "action" : torch.empty((buffer_size), dtype=torch.float32, requires_grad=False),
            "reward" : torch.empty((buffer_size), dtype=torch.float32, requires_grad=False),
            "next_state" : torch.empty((buffer_size,3), dtype=torch.float32, requires_grad=False),
            "trunc" : torch.empty((buffer_size), dtype=torch.bool, requires_grad=False)
        }
        self._rng = np.random.default_rng(seed=seed)

        self._buffer_size = buffer_size
        self._idx = 0
        self._full = False

    def addTransition(self, state, action ,reward, next_state, trunc):
        # if len(self._buffer) >= self._buffer_size:
        #     self._buffer.pop(0)
        # self._buffer.append((state, action, reward, next_state, trunc))
        self._buffer["state"][self._idx,:] = self.numpy2tensor(state)
        self._buffer["action"][self._idx] = self.numpy2tensor(action)
        self._buffer["reward"][self._idx] = self.numpy2tensor(reward)
        self._buffer["next_state"][self._idx,:] = self.numpy2tensor(next_state)
        self._buffer["trunc"][self._idx] = self.numpy2tensor(trunc)

        # increment index
        self._idx = self._idx + 1
        if self._idx >= self._buffer_size:
            self._idx = 0
            self._full = True

    def sampleBatch(self, batch_size):
        
        # return False if buffer is not full
        if not self._full:
            return None

        # raise error if batch size is larger than buffer size
        if batch_size > self._buffer_size:
            raise ValueError("Batch size is larger than maximum buffer size")       

        # choose random samples
        rand_idx = self._rng.choice(self._buffer_size, size=batch_size, replace=False) # TODO: should the samples be removed ???
        batch = { 
            "state" : self._buffer["state"][rand_idx,:],
            "action" : self._buffer["action"][rand_idx],
            "reward" : self._buffer["reward"][rand_idx],
            "next_state" : self._buffer["next_state"][rand_idx,:],
            "trunc" : self._buffer["trunc"][rand_idx]
        }
        return batch
    
    def numpy2tensor(self, array):
        if not torch.is_tensor(array):
            return torch.tensor(array, dtype=torch.float32, requires_grad=False)
        else:
            return array
        
    def detachClone(self, batch):
        if batch is not None:
            batch = { 
                "state" : batch["state"].clone().detach(),
                "action" : batch["action"].clone().detach(),
                "reward" : batch["reward"].clone().detach(),
                "next_state" : batch["next_state"].clone().detach(),
                "trunc" : batch["trunc"].clone().detach()
            }
        return batch
    

def testReplayBuffer():
    buffer = ReplayBuffer(max_size=10, seed=1)
    for i in range(30):
        buffer.addTransition(state=np.array([1,1,1])*i, action=np.array([1])*i, reward=1*i, next_state=np.array([1,1,1])*i, trunc=False)

        sample = buffer.sampleBatch(batch_size=5)
        print(sample)


if __name__ == "__main__":
    testReplayBuffer()