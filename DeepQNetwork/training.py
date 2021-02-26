import numpy as np
import random
from time import time

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.losses import MeanSquaredError
from keras.optimizers import Adam, SGD

from DeepQNetwork.environment import Maze
from DeepQNetwork.experience import Experience

# Create Maze
mazePath = '../maze_pictures/3x3maze.png'
maze = Maze(mazePath)
num_actions = maze.numActions

# Initialize Model
lr = 0.001
model = Sequential()
model.add(Flatten(input_shape=(maze.freeCellsState.shape[1],)))
model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(num_actions))

model.compile(optimizer=SGD(learning_rate=lr), loss=MeanSquaredError())
print(model.summary())

# initialize Experience
experience = Experience(maxSize=1000, discount=0.9)
minReward = -0.5 * maze.maze.size
epsilon = 1
epsilonDecayRate = 0.995
maxEpochs = 100

# training
epoch = 0
running = True
while running:
    epoch += 1
    # startPos = random.choice(maze.getFreeCells())
    startPos = maze.startPos
    maze.reset(startPos)
    totalReward = 0
    currentEnvState = maze.freeCellsState
    status = maze.status
    nbmoves = 0
    t0 = time()
    while status != 1 and totalReward > minReward:
        if random.random() <= epsilon:
            direction = random.randint(0, 3)
        else:
            direction = np.argmax(model.predict(currentEnvState))

        nextEnvState, reward, status = maze.move(direction)
        totalReward += reward

        experience.remember([currentEnvState, direction, reward, nextEnvState, status])
        nbmoves += 1

        if (nbmoves % 8 == 0 or status == 1) and len(experience.memory) > 16:
            inputs, targets = experience.createBatch(model, BatchSize=16)
            history = model.fit(inputs, targets, epochs=8, batch_size=16, verbose=0)

        currentEnvState = nextEnvState

    print('Epoch: %d, nb moves: %d, epsilon: %.5f, Win: %d, duration: %.2f' % (epoch, nbmoves, epsilon, status, time()-t0))

    epsilon *= epsilonDecayRate
