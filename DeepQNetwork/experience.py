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

    def createBatch(self, model, BatchSize: int) -> tuple:
        batchSize = min(BatchSize, len(self.memory))

        sample = random.sample(self.memory, batchSize)
        currentStates = np.array([mem[0][0] for mem in sample])
        currentTargets = model.predict(currentStates)
        nextStates = np.array([mem[3][0] for mem in sample])
        nextTargets = model.predict(nextStates)

        for idx, (currentEnvState, direction, reward, nextEnvState, status) in enumerate(sample):
            if status == 1:
                currentTargets[idx, direction] = reward
            else:
                currentTargets[idx, direction] = reward + self.discount * np.max(nextTargets[idx])

        return currentStates, currentTargets
