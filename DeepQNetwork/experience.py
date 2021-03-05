import random


class Experience():
    def __init__(self, maxSize: int) -> None:
        self.maxSize = maxSize
        self.memory = list()

    def remember(self, state: list) -> None:
        self.memory.append(state)
        if len(self.memory) > self.maxSize:
            del self.memory[0]

    def createBatch(self, BatchSize: int) -> tuple:
        batchSize = min(BatchSize, len(self.memory))
        sample = random.sample(self.memory, batchSize)

        return sample
