import numpy as np
import random
from time import time

from DeepQNetwork.environment import Maze
from DeepQNetwork.experience import Experience
from DeepQNetwork.experience import Agent

# Create Maze
mazePath = '../maze_pictures/3x3maze.png'
maze = Maze(mazePath)

# initialize Experience
experience = Experience(maxSize=1000, discount=0.9)
minReward = -maze.maze.size
epsilon = 1
epsilonDecayRate = 0.01
maxEpochs = 100

model = Agent(maze).model
target_model = Agent(maze).model
target_model.set_weights(model.get_weights())

# training

target_update_counter = 0
epoch = 0
running = True
tstart_training = time()
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
        target_update_counter += 1

        if random.random() <= epsilon:
            direction = random.randint(0, 3)
        else:
            direction = np.argmax(model.predict(currentEnvState))

        nextEnvState, reward, status = maze.move(direction)
        totalReward += reward

        experience.remember([currentEnvState, direction, reward, nextEnvState, status])
        nbmoves += 1

        if (nbmoves % 4 == 0 or status == 1) and len(experience.memory) >= 1000:
            inputs, targets = experience.createBatch(model, target_model, BatchSize=64)
            history = model.fit(inputs, targets, epochs=8, batch_size=64, verbose=0)

        if target_update_counter % 100 == 0:
            target_model.set_weights(model.get_weights())

        currentEnvState = nextEnvState

    print('Epoch: %d, epsilon: %.5f, nb moves: %d, totReward: %.2f, Win: %d, duration: %.2f'
          % (epoch, epsilon, nbmoves, totalReward, status, time()-t0))

    epsilon = np.exp(-epsilonDecayRate * epoch)

target_model.save('../test.h5')
print("ready to try game. Trinaing done in %.1f secs" % (time()-tstart_training))
