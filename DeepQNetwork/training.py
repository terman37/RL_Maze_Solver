import numpy as np
import random
from time import time

from DeepQNetwork.environment import Maze
from DeepQNetwork.experience import Experience
from DeepQNetwork.agent import Agent

# Create Maze
mazePath = '../maze_pictures/2x2maze.png'
maze = Maze(mazePath)

# initialize Experience
experience = Experience(maxSize=8*maze.maze.size, discount=0.9)
minReward = - maze.maze.size
epsilon = 1
minEpsilon = 0.1
discount = 0.9

maxEpochs = 100

agent = Agent(maze)
target_agent = Agent(maze)
target_agent.model.set_weights(agent.model.get_weights())
BS = 8

# training
maze.initDisplay()

target_update_counter = 0
epoch = 0
running = True
tstart_training = time()
while epoch < maxEpochs:
    epoch += 1
    startPos = maze.startPos
    maze.reset(startPos)
    totalReward = 0
    currentEnvState = maze.freeCellsState
    nbmoves = 0
    t0 = time()
    
    while maze.status == 0 and totalReward > minReward:
        maze.display_move()
        target_update_counter += 1

        possibleDirs = maze.getPossibleDirections(maze.currentPos)
        if maze.status == 0:
            if random.random() <= epsilon:
                direction = random.choice(possibleDirs)
            else:
                qvalues = agent.predict_qvalues(currentEnvState)[0]
                _, direction = agent.get_max(possibleDirs, qvalues)

            nextEnvState, reward, status = maze.move(direction)
            totalReward += reward

            experience.remember([currentEnvState, direction, reward, nextEnvState, status, possibleDirs])
            nbmoves += 1

        if (nbmoves % 8 == 0 or maze.status != 0) and len(experience.memory) > BS:
            batch = experience.createBatch(BatchSize=BS)

            currentStates = np.array([mem[0][0] for mem in batch])
            currentTargets = agent.predict_qvalues(currentStates)
            nextStates = np.array([mem[3][0] for mem in batch])
            nextTargets = target_agent.predict_qvalues(nextStates)

            for idx, (currentEnvState, direction, reward, nextEnvState, status, possibleDirs) in enumerate(batch):
                if status == 1:
                    currentTargets[idx, direction] = reward
                else:
                    maxQ, _ = agent.get_max(possibleDirs, nextTargets[idx])
                    currentTargets[idx, direction] = reward + discount * maxQ

                history = agent.model.fit(x=currentStates[idx].reshape(1,-1),
                                          y=currentTargets[idx].reshape(1,-1),
                                          epochs=8,
                                          verbose=0)

        if target_update_counter % 1 == 0:
            target_agent.model.set_weights(agent.model.get_weights())

        currentEnvState = nextEnvState

    w = maze.watch_all_free_cells()
    q = np.round(agent.model.predict(w), 2)
    q = [list(i) for i in q]

    for i in zip(w.tolist(), q):
        print(i)

    print('Epoch: %d, epsilon: %.5f, nb moves: %d, totReward: %.2f, Win: %d, duration: %.2f'
          % (epoch, epsilon, nbmoves, totalReward, maze.status, time()-t0))

    epsilon = np.exp(np.log(minEpsilon)/maxEpochs * epoch)

agent.model.save('./test.h5')
print("ready to try game. Training done in %.1f secs" % (time()-tstart_training))


startPos = maze.startPos
maze.reset(startPos)
currentEnvState = maze.freeCellsState
nbmoves = 0

maze.initDisplay()
while maze.status == 0:
    possibleDirs = maze.getPossibleDirections(maze.currentPos)
    qvalues = agent.predict_qvalues(currentEnvState)[0]
    _, direction = agent.get_max(possibleDirs, qvalues)
    nextEnvState, reward, status = maze.move(direction)
    currentEnvState = nextEnvState
    maze.display_move()
