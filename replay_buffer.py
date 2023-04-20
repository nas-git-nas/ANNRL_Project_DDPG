import numpy as np



class ReplayBuffer():
    def __init__(self, max_size, seed):
        self._max_size = max_size
        self._buffer = []
        self._rng = np.random.default_rng(seed=seed)

    def addTransition(self, state, action ,reward, next_state, trunc):
        if len(self._buffer) >= self._max_size:
            self._buffer.pop(0)
        self._buffer.append((state, action, reward, next_state, trunc))

    def sampleBatch(self, batch_size):
        if batch_size > len(self._buffer):
            raise ValueError("Batch size is larger than buffer size")
        
        batch = self._rng.choice(np.array(self._buffer), size=batch_size, replace=False) # TODO: should the samples be removed ???
        return batch
    

def testReplayBuffer():
    buffer = ReplayBuffer(max_size=10, seed=1)
    for i in range(15):
        buffer.addTransition(i, i, i, i, i)
    print(buffer.sampleBatch(batch_size=5))


if __name__ == "__main__":
    testReplayBuffer()