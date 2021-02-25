import numpy as np
# from DQN.environment import Environment
from DQN.maze import Maze
from DQN.experienceMemory import Memory
from DQN.experienceMemory import Model

# Environment
mazePath = '../maze_pictures/3x3maze.png'
maze = Maze(mazePath)
# env = Environment(maze)

# ExperienceReplay
mem = Memory(1000, 0.9)
brain = Model(2, 4, 0.001)

# Experience Parameter
minReward = -0.5 * maze.maze.size / 2

epsilon = 1  # exploration factor
epsilonDecayRate = 0.995


# Running experience
epoch = 0
running = True
while running:
    # one experience
    epoch += 1
    continueGame = True
    win = False
    currentPos = maze.startPos
    totalReward = 0
    visited = list()
    visited.append(currentPos)
    # env.displayMove(currentPos)
    while continueGame:
        if np.random.rand() <= epsilon:
            possibleDirs = maze.possibleDirections(currentPos, visited)
            if len(possibleDirs) != 0:
                direction = np.random.choice(possibleDirs)
            else:
                continueGame = False
        else:
            qvalues = brain.model.predict(np.array(currentPos).reshape((1, 2)))[0]
            direction = np.argmax(qvalues)

        if continueGame:
            nextPos, reward = maze.move(currentPos, direction)

            if nextPos in visited:
                reward += -0.5
            else:
                visited.append(nextPos)

            totalReward += reward

            if currentPos == maze.finishPos:
                win = True

            mem.remember([np.array(currentPos).reshape((1, 2)),
                          direction,
                          reward,
                          np.array(nextPos).reshape((1, 2))],
                         win)
            inputs, targets = mem.createBatch(brain.model, 16)
            brain.model.train_on_batch(inputs, targets)

            currentPos = nextPos
            # env.displayMove(currentPos)
            # env.displayText(str(round(totalReward, 2)))
            if totalReward < minReward or win:
                continueGame = False

    epsilon *= epsilonDecayRate
    print('Epoch: %d Epsilon: %.5f Total Reward: %.2f' % (epoch, epsilon, totalReward))

print('end')
