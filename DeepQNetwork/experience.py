import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras import initializers


class Experience():
    def __init__(self, maxSize: int, discount: float) -> None:
        self.maxSize = maxSize
        self.discount = discount
        self.memory = list()

    def remember(self, state: list) -> None:
        self.memory.append(state)
        if len(self.memory) > self.maxSize:
            del self.memory[0]

    def createBatch(self, model, target_model, BatchSize: int) -> tuple:
        batchSize = min(BatchSize, len(self.memory))

        sample = random.sample(self.memory, batchSize)
        currentStates = np.array([mem[0][0] for mem in sample])
        currentTargets = model.predict(currentStates)
        nextStates = np.array([mem[3][0] for mem in sample])
        nextTargets = target_model.predict(nextStates)

        for idx, (currentEnvState, direction, reward, nextEnvState, status) in enumerate(sample):
            if status == 1:
                currentTargets[idx, direction] = reward
            else:
                currentTargets[idx, direction] = reward + self.discount * np.max(nextTargets[idx])

        return currentStates, currentTargets


class Agent():
    def __init__(self, maze):
        lr = 0.001
        num_actions = maze.numActions
        init = initializers.RandomUniform(minval=-0.01, maxval=0.01)
        model = Sequential()
        model.add(Flatten(input_shape=(maze.freeCellsState.shape[1],)))
        model.add(Dense(maze.maze.size,
                        kernel_initializer=init,
                        activation='relu'))
        # model.add(Dense(16, activation='relu'))
        model.add(Dense(num_actions,
                        kernel_initializer=init))

        model.compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError())
        # print(model.summary())
        self.model = model
