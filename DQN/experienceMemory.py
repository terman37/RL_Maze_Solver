from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np


class Memory():
    def __init__(self, maxSize, discount) -> None:
        self.maxSize = maxSize
        self.discount = discount
        self.experiences = list()

    def remember(self, step, win):
        self.experiences.append([step, win])
        if len(self.experiences) > self.maxSize:
            del self.experiences[0]

    def createBatch(self, model, maxBatchSize):
        batchSize = min(maxBatchSize, len(self.experiences))
        inputs = np.zeros((batchSize, 2))
        targets = np.zeros((batchSize, 4))

        selection = np.random.randint(0, len(self.experiences), size=batchSize)
        for i, idx in enumerate(selection):
            currentPos, direction, reward, nextPos = self.experiences[idx][0]
            win = self.experiences[idx][1]
            inputs[i] = currentPos
            targets[i] = model.predict(currentPos)[0]
            if win:
                targets[i][direction] = reward
            else:
                targets[i][direction] = reward + self.discount * np.max(model.predict(nextPos)[0])

        return inputs, targets


class Model():
    def __init__(self, numInputs, numOutputs, lr):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.learningRate = lr

        # Creating the neural network
        self.model = Sequential()
        self.model.add(Dense(units=32, activation='relu', input_shape=(self.numInputs, )))
        self.model.add(Dense(units=16, activation='relu'))
        self.model.add(Dense(units=self.numOutputs))
        self.model.compile(optimizer=Adam(lr=self.learningRate), loss='mean_squared_error')
