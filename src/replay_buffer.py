import numpy as np
import random



class ReplayBuffer():
    def __init__(self, buffer_size, seed):
        self._buffer_size = buffer_size
        self._buffer = { 
            "state" : np.empty((buffer_size,3), dtype=float),
            "action" : np.empty((buffer_size), dtype=float),
            "reward" : np.empty((buffer_size), dtype=float),
            "next_state" : np.empty((buffer_size,3), dtype=float),
            "trunc" : np.empty((buffer_size), dtype=bool)
        }
        self._rng = np.random.default_rng(seed=seed)
        # random.seed(seed)

        self._idx = 0
        self._full = False

    def addTransition(self, state, action ,reward, next_state, trunc):
        # if len(self._buffer) >= self._buffer_size:
        #     self._buffer.pop(0)
        # self._buffer.append((state, action, reward, next_state, trunc))
        self._buffer["state"][self._idx,:] = state
        self._buffer["action"][self._idx] = action
        self._buffer["reward"][self._idx] = reward
        self._buffer["next_state"][self._idx,:] = next_state
        self._buffer["trunc"][self._idx] = trunc

        # increment index
        self._idx = self._idx + 1
        if self._idx >= self._buffer_size:
            self._idx = 0
            self._full = True

    def sampleBatch(self, batch_size):
        
        
        # # convert nested buffer list to numpy array
        # buffer = np.array(self._buffer, dtype=object)
        
        max_size = self._buffer_size
        if not self._full:
            max_size = self._idx

        # raise error if batch size is larger than buffer size
        if batch_size > self._buffer_size:
            raise ValueError("Batch size is larger than maximum buffer size")

        # # return False if buffer is smaller than batch size
        # if batch_size >= max_size:
        #     return False
        
        # return False if buffer is not full
        if not self._full:
            return False
        
        rand_idx = self._rng.choice(max_size, size=batch_size, replace=False) # TODO: should the samples be removed ???
        batch = { 
            "state" : self._buffer["state"][rand_idx,:],
            "action" : self._buffer["action"][rand_idx],
            "reward" : self._buffer["reward"][rand_idx],
            "next_state" : self._buffer["next_state"][rand_idx,:],
            "trunc" : self._buffer["trunc"][rand_idx]
        }
        # batch = self._rng.choice(self._buffer, size=batch_size, replace=False) 
        # batch = random.sample(self._buffer, batch_size)
        return batch
    

def testReplayBuffer():
    buffer = ReplayBuffer(max_size=10, seed=1)
    for i in range(30):
        buffer.addTransition(state=np.array([1,1,1])*i, action=np.array([1])*i, reward=1*i, next_state=np.array([1,1,1])*i, trunc=False)

        sample = buffer.sampleBatch(batch_size=5)
        print(sample)


if __name__ == "__main__":
    testReplayBuffer()