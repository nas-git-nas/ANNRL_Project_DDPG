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
        self._buffer["state"][self._idx,:] = self.numpy2tensor(state, shape_type="state")
        self._buffer["action"][self._idx] = self.numpy2tensor(action, shape_type="action")
        self._buffer["reward"][self._idx] = self.numpy2tensor(reward, shape_type="reward")
        self._buffer["next_state"][self._idx,:] = self.numpy2tensor(next_state, shape_type="next_state")
        self._buffer["trunc"][self._idx] = self.numpy2tensor(trunc, shape_type="trunc")

        # increment index
        self._idx = self._idx + 1
        if self._idx >= self._buffer_size:
            self._idx = 0
            self._full = True

    def sampleBatch(self, batch_size):
        
        # determine current buffer size
        if self._full:
            current_buffer_size = self._buffer_size
        else:
            current_buffer_size = self._idx

        # return None if batch_size is larger than current buffer size
        if batch_size > current_buffer_size:
            return None  

        # choose random samples
        rand_idx = self._rng.choice(current_buffer_size, size=batch_size, replace=False) # TODO: should the samples be removed ???
        batch = { 
            "state" : self._buffer["state"][rand_idx,:],
            "action" : self._buffer["action"][rand_idx],
            "reward" : self._buffer["reward"][rand_idx],
            "next_state" : self._buffer["next_state"][rand_idx,:],
            "trunc" : self._buffer["trunc"][rand_idx]
        }
        return batch
    
    def numpy2tensor(self, array, shape_type):
        # convert numpy array to tensor if necessary
        if not torch.is_tensor(array):
            tensor = torch.tensor(array, dtype=torch.float32, requires_grad=False)
        else:
            tensor = array

        # reshape tensor
        if shape_type == "state":
            tensor = tensor.reshape((-1,3))
        elif shape_type == "action":
            tensor = tensor.reshape((-1,1))
        elif shape_type == "reward":
            tensor = tensor.reshape((-1,1))
        elif shape_type == "next_state":
            tensor = tensor.reshape((-1,3))
        elif shape_type == "trunc":
            tensor = tensor.reshape((-1,1))
        else:
            raise ValueError("Invalid shape_type: {}".format(shape_type))
        
        return tensor
        
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