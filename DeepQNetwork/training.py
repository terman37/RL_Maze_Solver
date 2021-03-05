import numpy as np
import random
from time import time

from DeepQNetwork.environment import Maze
from DeepQNetwork.experience import Experience
from DeepQNetwork.agent import Agent

# HYPERPARAMETERS
BS = 8
NBMOVES_UPDATEMODEL = 8
COUNT_UPDATETARGET = 20

DISCOUNT = 0.8

MAXEPOCHS = 1000
STARTEPSILON = 1
MINEPSILON = 0.2
EPSILONDECAYRATE = 0.99


def training(maze, show=False):

    # Initialize agents
    agent = Agent(maze)
    target_agent = Agent(maze)
    target_agent.model.set_weights(agent.model.get_weights())

    # initialize Experience
    experience = Experience(maxSize=maze.maze.size)

    # minReward = - maze.maze.size
    epsilon = STARTEPSILON
    target_update_counter = 0
    epoch = 0

    tstart_training = time()
    while epoch < MAXEPOCHS:

        epoch += 1
        startPos = maze.startPos
        maze.reset(startPos)
        totalReward = 0
        currentEnvState = maze.freeCellsState
        nbmoves = 0
        t0 = time()

        while maze.status == 0:
            target_update_counter += 1

            possibleDirs = maze.getPossibleDirections(maze.currentPos)
            if maze.status == 0:
                if random.random() <= epsilon:
                    direction = random.choice(possibleDirs)
                else:
                    qvalues = agent.predict_qvalues(currentEnvState)[0]
                    _, direction = agent.get_max(possibleDirs, qvalues)

                nextEnvState, reward, status = maze.move(direction)
                curpos = np.where(currentEnvState == 1)[1][0]
                nextpos = np.where(nextEnvState == 1)[1][0]
                totalReward += reward

                experience.remember([currentEnvState, direction, reward, nextEnvState, status, possibleDirs, curpos, nextpos])
                nbmoves += 1

            if (nbmoves % NBMOVES_UPDATEMODEL == 0 or maze.status != 0) and len(experience.memory) >= BS:
                batch = experience.createBatch(BatchSize=BS)

                currentStates = np.asarray([mem[0][0] for mem in batch])
                currentTargets = agent.predict_qvalues(currentStates)
                nextStates = np.asarray([mem[3][0] for mem in batch])
                nextTargets = target_agent.predict_qvalues(nextStates)

                for idx, (currentEnvState, direction, reward, nextEnvState, status, possibleDirs, curpos, nextpos) in enumerate(batch):
                    for i in range(maze.numActions):
                        if i in possibleDirs:
                            if status != 0:
                                currentTargets[idx, direction] = reward
                            else:
                                maxQ, _ = agent.get_max(possibleDirs, nextTargets[idx])
                                currentTargets[idx, direction] = reward + DISCOUNT * maxQ
                        else:
                            currentTargets[idx, i] = 0

                _ = agent.model.fit(x=currentStates, y=currentTargets, epochs=8, verbose=0)

                print('batch')
                for line in list(zip(np.where(currentStates==1)[1].tolist(), [[round(l,3) for l in x] for x in currentTargets.tolist()])):
                    print(line)
                print('batchwait')

            if target_update_counter % COUNT_UPDATETARGET == 0:
                target_agent.model.set_weights(agent.model.get_weights())

            currentEnvState = nextEnvState

        if show:
            maze.display_move(50)

        postowatch = [5, 31, 42, 55]
        for pos in postowatch:
            st = np.zeros(currentEnvState.shape)
            st[0, pos] = 1
            q = agent.predict_qvalues(st)
            print('pos:', pos, ' -> ', q)

        print('Epoch: %d, epsilon: %.5f, nb moves: %d, totReward: %.2f, Win: %d, duration: %.2f'
              % (epoch, epsilon, nbmoves, totalReward, maze.status, time()-t0))

        # early stopping if training is enough
        if maze.status == 1:
            res = playMaze(maze, agent)
            # epsilon = np.exp(np.log(MINEPSILON)/MAXEPOCHS * epoch)

            if res == 1:
                print('solution ok')
                break
            else:
                print('continue training')
            
        if epsilon > MINEPSILON:
            epsilon *= EPSILONDECAYRATE
            
    target_agent.model.set_weights(agent.model.get_weights())
    # target_agent.model.save('./test.h5')
    print("ready to try game. Training done in %.1f secs" % (time()-tstart_training))
    return target_agent


def playMaze(maze, model, show=False):
    startPos = maze.startPos
    maze.reset(startPos)
    currentEnvState = maze.freeCellsState

    while maze.status == 0:
        possibleDirs = maze.getPossibleDirections(maze.currentPos)
        qvalues = model.predict_qvalues(currentEnvState)[0]
        _, direction = model.get_max(possibleDirs, qvalues)
        nextEnvState, reward, status = maze.move(direction)
        currentEnvState = nextEnvState
        if show:
            maze.display_move(200)

    return maze.status


if __name__ == "__main__":
    # Create Maze
    mazePath = '../maze_pictures/6x6maze2.png'
    maze = Maze(mazePath)
    maze.initDisplay()

    # train and get trained model
    model = training(maze, show=True)

    # play Maze with trained model
    playMaze(maze, model, show=True)

    print('TADA !')
