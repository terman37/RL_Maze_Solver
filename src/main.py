import numpy as np
from environment import Environment
import pygame as pg
from numba import jit
from time import time


@jit(nopython=True)
def training(epochsNb, rewardBoard):
    # Algorithm parameters
    gamma = 0.9
    alpha = 0.75
    # Allowed positions on board
    allowedPos = np.where(rewardBoard.sum(axis=0) != 0)[0]
    qTable = rewardBoard.copy()
    for _ in range(epochsNb):
        # select a random allowed position
        startingPos = np.random.choice(allowedPos)
        # possible destinations
        possibleDest = np.where(rewardBoard[startingPos] != 0)[0]
        # play a random action
        destPos = np.random.choice(possibleDest)
        reward = rewardBoard[startingPos][destPos]
        # update QValue
        maxQValue = qTable[destPos][qTable[destPos].nonzero()].max()
        TempDiff = reward + gamma * maxQValue - qTable[startingPos][destPos]
        qTable[startingPos][destPos] += alpha * TempDiff
    return qTable


# Initial state
mazePath = './maze_pictures/20x30maze.png'
env = Environment(mazePath)
env.displayMaze()
env.displayText('Training...')

# Training
tstart = time()
qTable = training(epochsNb=2000000, rewardBoard=env.rewardBoard)
tfinish = time()
print("Training Duration %.2f secs" % (tfinish-tstart))

# Showing solution
env.displayText('Showing solution')

startPos = env.startPos[0] * env.mazeColNb + env.startPos[1]
finishPos = env.finishPos[0] * env.mazeColNb + env.finishPos[1]
currentPos = startPos
env.moveInMaze(currentPos)


running = True
while running:
    if currentPos != finishPos:
        action = np.where(qTable[currentPos] == qTable[currentPos][qTable[currentPos].nonzero()].max())[0][0]
        currentPos = action
        env.moveInMaze(currentPos)
    else:
        env.displayText('Tada !')

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
