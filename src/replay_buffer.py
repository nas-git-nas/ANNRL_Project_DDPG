import numpy as np
import random



class ReplayBuffer():
    def __init__(self, max_size, seed):
        self._max_size = max_size
        self._buffer = { 
            "state" : np.empty((max_size,3), dtype=float),
            "action" : np.empty((max_size), dtype=float),
            "reward" : np.empty((max_size), dtype=float),
            "next_state" : np.empty((max_size,3), dtype=float),
            "trunc" : np.empty((max_size), dtype=bool)
        }
        self._rng = np.random.default_rng(seed=seed)
        # random.seed(seed)

        self._idx = 0
        self._full = False

    def addTransition(self, state, action ,reward, next_state, trunc):
        # if len(self._buffer) >= self._max_size:
        #     self._buffer.pop(0)
        # self._buffer.append((state, action, reward, next_state, trunc))
        self._buffer["state"][self._idx,:] = state
        self._buffer["action"][self._idx] = action
        self._buffer["reward"][self._idx] = reward
        self._buffer["next_state"][self._idx,:] = next_state
        self._buffer["trunc"][self._idx] = trunc

        # increment index
        self._idx = self._idx + 1
        if self._idx >= self._max_size:
            self._idx = 0
            self._full = True

    def sampleBatch(self, batch_size):
        if batch_size > self._max_size:
            raise ValueError("Batch size is larger than maximum buffer size")
        
        # # convert nested buffer list to numpy array
        # buffer = np.array(self._buffer, dtype=object)
        
        idx_max = self._max_size
        if not self._full:
            idx_max = self._idx

        # return False if buffer is smaller than batch size
        if batch_size >= idx_max:
            return False
        
        rand_idx = self._rng.choice(idx_max, size=batch_size, replace=False)
        batch = { 
            "state" : self._buffer["state"][rand_idx,:],
            "action" : self._buffer["action"][rand_idx],
            "reward" : self._buffer["reward"][rand_idx],
            "next_state" : self._buffer["next_state"][rand_idx,:],
            "trunc" : self._buffer["trunc"][rand_idx]
        }
        # batch = self._rng.choice(self._buffer, size=batch_size, replace=False) # TODO: should the samples be removed ???
        # batch = random.sample(self._buffer, batch_size)
        return batch
    

def testReplayBuffer():
    buffer = ReplayBuffer(max_size=10, seed=1)
    for i in range(15):
        buffer.addTransition(state=np.array([1,2,3])*i, action=np.array([4])*i, reward=5*i, next_state=np.array([1,2,3])*2*i, trunc=False)

    sample = buffer.sampleBatch(batch_size=5)
    print(sample)


if __name__ == "__main__":
    testReplayBuffer()