import numpy as np
import random


class Experience():
    def __init__(self, maxSize: int, discount: float) -> None:
        self.maxSize = maxSize
        self.discount = discount
        self.memory = list()

    def remember(self, state: list) -> None:
        self.memory.append(state)
        if len(self.memory) > self.maxSize:
            del self.memory[0]

    def createBatch(self, BatchSize: int) -> tuple:
        batchSize = min(BatchSize, len(self.memory))

        sample = random.sample(self.memory, batchSize)

        return sample
