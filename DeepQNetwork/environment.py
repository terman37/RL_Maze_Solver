from PIL import Image
import numpy as np
import pygame as pg


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ONGOING, WIN, LOSE = 0, 1, 2

WALL, FREE, VISITED, FINISH, CURRENTPOS = 0, 1, 2, 3, 4
MAZE_COLORS = np.array([[0,0,0], [255,255,255], [127,255,255], [0,255,0], [0,0,255]])


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
        self._maze = mazeArray.astype('int32')

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
            reward = -0.9
        elif self.maze[row, col] == WALL:
            reward = -0.75
        elif self.maze[row, col] == FREE:
            reward = -0.03
        elif self.maze[row, col] == VISITED:
            reward = -0.25
        elif self.maze[row, col] == FINISH:
            reward = 1
            self.status = WIN

        if len(self.getPossibleDirections(position)) == 0 and self.status != WIN:
            reward = -1
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
        # freeCells = list()
        # for row in range(self.mazeRowNb):
        #     for col in range(self.mazeColNb):
        #         if self.maze[row, col] != WALL:
        #             freeCells.append(self.maze[row, col])
        # return np.array(freeCells).reshape(1, -1)
        return np.array(self.currentPos).reshape(1, -1)

    def watch_all_free_cells(self):
        freeCells = list()
        for row in range(self.mazeRowNb):
            for col in range(self.mazeColNb):
                if self.maze[row, col] != WALL:
                    freeCells.append(np.array([row, col]))
        return np.array(freeCells).reshape(len(freeCells), -1)

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
        surface = pg.surfarray.make_surface(MAZE_COLORS[np.transpose(self.maze)])
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

    def display_move(self):
        self.displayMaze()
        pg.time.wait(100)
        pg.event.pump()
