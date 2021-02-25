import numpy as np
from QTableLearning.environment import Environment
from numba import jit
from time import time
import argparse


@jit(nopython=True)
def initRewardBoard(maze):
    # Rewards:
    rewards = [0, -0.03, 1]
    # wall = move not allowed =0
    # living penalty = -0.03
    # finish reward = 1

    rowNb = maze.shape[0]
    colNb = maze.shape[1]
    rb = np.zeros((rowNb, colNb, 4))

    for row in range(rowNb):
        for col in range(colNb):
            # if not a wall
            # directions: 0=up, 1=down, 2=left, 3=right
            if maze[row, col] != 0:
                if row > 0:
                    rb[row, col, 0] = rewards[maze[row-1, col]]
                if row < rowNb-1:
                    rb[row, col, 1] = rewards[maze[row+1, col]]
                if col > 0:
                    rb[row, col, 2] = rewards[maze[row, col-1]]
                if col < colNb-1:
                    rb[row, col, 3] = rewards[maze[row, col+1]]
    return rb


@jit(nopython=True)
def getDestination(position, direction):
    if direction == 0:  # up
        destination = position[0]-1, position[1]
    if direction == 1:  # down
        destination = position[0]+1, position[1]
    if direction == 2:  # left
        destination = position[0], position[1]-1
    if direction == 3:  # right
        destination = position[0], position[1]+1
    return destination


@jit(nopython=True)
def training(epochsNb, maze, rb):
    # Algorithm parameters
    gamma = 0.9
    alpha = 0.75
    # Allowed positions on board
    allowedPos = np.where(maze != 0)
    qTable = rb.copy()
    for _ in range(epochsNb):
        # select a random allowed position
        idx = np.random.randint(0, len(allowedPos[0]))
        startingPos = allowedPos[0][idx], allowedPos[1][idx]
        # possible directions
        possibleDirections = rb[startingPos].nonzero()[0]
        # play a random action
        selectedDirection = np.random.choice(possibleDirections)
        DestinationPos = getDestination(startingPos, selectedDirection)
        # update QValue
        reward = rb[startingPos][selectedDirection]
        maxQValue = qTable[DestinationPos][qTable[DestinationPos].nonzero()].max()
        TempDiff = reward + gamma * maxQValue - qTable[startingPos][selectedDirection]
        qTable[startingPos][selectedDirection] += alpha * TempDiff
    return qTable


# Cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mazefilepath', type=str, default='../maze_pictures/20x20maze.png', help='path to maze file')
args = parser.parse_args()

# Load Environment
mazePath = args.mazefilepath
env = Environment(mazePath)
env.displayMaze()

# Training
env.displayText('Training...')
tstart = time()
rewardBoard = initRewardBoard(env.maze)
qTable = training(epochsNb=10000000, maze=env.maze, rb=rewardBoard)
tfinish = time()
print("Training Duration %.2f secs" % (tfinish-tstart))

# Showing solution
env.displayText('Showing solution')

currentPos = env.startPos
env.displayMove(currentPos)

running = True
while running:
    if currentPos != env.finishPos:
        direction = np.argmax(np.where(qTable[currentPos] != 0, qTable[currentPos], -np.inf))
        currentPos = getDestination(currentPos, direction)
        env.displayMove(currentPos)
    else:
        env.displayText('Tada !')
        running = env.wait_and_quit()
