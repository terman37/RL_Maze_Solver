import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras import initializers


class Agent():
    def __init__(self, maze):
        lr = 0.001
        num_actions = maze.numActions
        init = initializers.RandomUniform(minval=-0.001, maxval=0.001)
        model = Sequential()
        model.add(Flatten(input_shape=(maze.freeCellsState.shape[1],)))
        model.add(Dense(maze.maze.size,
                        kernel_initializer=init,
                        activation='relu'))
        # model.add(Dense(32,
        #                 kernel_initializer=init,
        #                 activation='relu'))
        model.add(Dense(num_actions,
                        kernel_initializer='zeros',
                        activation='linear'))

        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        # print(model.summary())
        self.model = model

    def predict_qvalues(self, states):
        return self.model.predict(states)

    def get_max(self, possibleDirs, qvalues):
        qvalues = qvalues
        maxQ = np.max(qvalues[possibleDirs])
        direction = np.where(qvalues == maxQ)[0][0]
        return maxQ, direction
