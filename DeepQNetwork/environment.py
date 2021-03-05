from PIL import Image
import numpy as np
import pygame as pg


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
POUT, PWALL, PVISITED, PFREE, PFINISH, PLOSE = -0.8, -0.75, -0.25, -0.01, 1, -1
ONGOING, WIN, LOSE = 0, 1, 2

WALL, FREE, VISITED, FINISH, CURRENTPOS = 0, 1, 0.25, 0.5, 0.75
CWALL, CFREE, CVISITED, CFINISH, CCURRENTPOS = [0, 0, 0], [255, 255, 255], [127, 255, 255], [0, 255, 0], [0, 0, 255]
# CWALL, CFREE, CVISITED, CFINISH, CCURRENTPOS = 0x0000000, 0xffffff00, 0x7fffff00, 0x00ff0000, 0x0000ff00


class Maze():
    def __init__(self, filePath) -> None:
        self.mazePath = filePath
        self.buildMaze()
        self.reset(self.startPos)
        self.numActions = 4  # UP/DOWN/LEFT/RIGHT

    def buildMaze(self) -> None:
        # load file and convert it to bilevel
        img = Image.open(self.mazePath)
        img2lvl = img.convert('1')

        # convert to simplified array
        # original image walls are 2px thick and cells are 14x14px
        npImg = np.asarray(img2lvl)
        self.originalImgSize = (npImg.shape[1], npImg.shape[0])

        self.mazeRowNb = int((npImg.shape[0]-2)/16)*2+1
        self.mazeColNb = int((npImg.shape[1]-2)/16)*2+1

        mazeArray = np.empty((self.mazeRowNb, self.mazeColNb))
        for row in range(0, self.mazeRowNb):
            for col in range(0, self.mazeColNb):
                mazeArray[row, col] = npImg[row * 8, col * 8]

        self.startPos = (0, mazeArray[0].nonzero()[0][0])
        self.finishPos = (mazeArray.shape[0]-1, mazeArray[-1].nonzero()[0][0])
        mazeArray[self.finishPos] = FINISH
        self._maze = mazeArray.astype('float32')
        # self._maze = mazeArray

    def reset(self, startPosition: tuple) -> None:
        self.maze = np.copy(self._maze)
        self.currentPos = startPosition
        self.markCurrentCell(self.currentPos)
        self.freeCellsState = self.getStateFreeCells()
        self.status = ONGOING

    def getNextPosition(self, position, direction):
        row, col = position
        if direction == UP:
            row -= 1
        if direction == DOWN:
            row += 1
        if direction == LEFT:
            col -= 1
        if direction == RIGHT:
            col += 1
        return (row, col)

    def move(self, direction: int) -> tuple:

        row, col = self.getNextPosition(self.currentPos, direction)
        nextPos = (row, col)

        self.markVisited(self.currentPos)
        reward = self.getreward(nextPos)

        self.markCurrentCell(nextPos)
        self.currentPos = nextPos
        self.freeCellsState = self.getStateFreeCells()

        return self.freeCellsState, reward, self.status

    def getreward(self, position: tuple) -> float:
        row, col = position

        if row < 0 or row >= self.mazeRowNb or col < 0 or col >= self.mazeColNb:  # OUT OF MAZE
            reward = POUT
        elif self.maze[row, col] == WALL:
            reward = PWALL
        elif self.maze[row, col] == FREE:
            reward = PFREE
        elif self.maze[row, col] == VISITED:
            reward = PVISITED
        elif self.maze[row, col] == FINISH:
            reward = PFINISH
            self.status = WIN

        if len(self.getPossibleDirections(position)) == 0 and self.status != WIN:
            reward = PLOSE
            self.status = LOSE

        return reward

    def checkValidMove(self, position: tuple) -> bool:
        row, col = position
        if (row < 0 or row >= self.mazeRowNb or col < 0 or col >= self.mazeColNb or
                self.maze[row, col] == WALL or
                self.maze[row, col] == VISITED):
            return False
        else:
            return True

    def getPossibleDirections(self, position):
        possibleDirs = list()
        for direction in range(self.numActions):
            row, col = self.getNextPosition(position, direction)
            if self.checkValidMove((row, col)):
                possibleDirs.append(direction)

        return possibleDirs

    def markVisited(self, position: tuple) -> None:
        if self.maze[position] != FINISH:
            self.maze[position] = VISITED

    def markCurrentCell(self, position: tuple) -> None:
        self.maze[position] = CURRENTPOS

    def getStateFreeCells(self):
        freeCells = list()
        for row in range(self.mazeRowNb):
            for col in range(self.mazeColNb):
                if self.maze[row, col] != WALL:
                    if self.maze[row, col] == CURRENTPOS:
                        freeCells.append(1)
                    else:
                        freeCells.append(0)

        return np.asarray(freeCells).reshape(1, -1)

    def initDisplay(self):
        # reshape factor to avoid app bigger than screen
        maxWidth = 900
        maxHeight = 900

        self.rFactor = 1
        if self.originalImgSize[0] > maxWidth or self.originalImgSize[1] > maxHeight:
            self.rFactor = round(min(maxWidth/self.originalImgSize[0], maxHeight/self.originalImgSize[1]), 2)

        # init app screen
        self.screen = pg.display.set_mode((int(self.rFactor * self.originalImgSize[0]+100),
                                           int(self.rFactor * self.originalImgSize[1]+100)))
        self.screen.fill([227, 227, 227])
        pg.display.set_caption('Maze Solver using Naive QLearning')

        # refresh
        pg.display.flip()

    def displayMaze(self):
        mazeDisp = np.copy(np.transpose(self.maze))

        mazeDispColors = np.zeros((mazeDisp.shape[0], mazeDisp.shape[1], 3))
        mazeDispColors[mazeDisp == WALL] = CWALL
        mazeDispColors[mazeDisp == FREE] = CFREE
        mazeDispColors[mazeDisp == VISITED] = CVISITED
        mazeDispColors[mazeDisp == CURRENTPOS] = CCURRENTPOS
        mazeDispColors[mazeDisp == FINISH] = CFINISH

        surface = pg.surfarray.make_surface(mazeDispColors)

        surface = pg.transform.scale(surface, tuple(int(i * self.rFactor) for i in self.originalImgSize))
        self.screen.blit(surface, (50, 50))

        # refresh
        pg.display.flip()

    def displayText(self, text):
        pg.draw.rect(self.screen, (227, 227, 227), (50, 10, 200, 30))
        myfont = pg.font.Font(pg.font.get_default_font(), 20)
        text_surface = myfont.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (55, 15))

        # refresh
        pg.display.flip()

    def display_move(self, waitTime=200):
        self.displayMaze()
        pg.time.wait(waitTime)
        pg.event.pump()
