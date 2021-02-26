from PIL import Image
import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
WALL, FREE, VISITED, FINISH = 0, 1, 2, 9
ONGOING, WIN = 0, 1


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
        self.markVisited(self.currentPos)
        self.freeCellsState = self.getStateFreeCells()
        self.status = ONGOING

    def move(self, direction: int) -> tuple:
        row, col = self.currentPos
        if direction == UP:
            row -= 1
        if direction == DOWN:
            row += 1
        if direction == LEFT:
            col -= 1
        if direction == RIGHT:
            col += 1

        reward = self.getreward((row, col))
        if self.checkValidMove((row, col)):
            nextPos = (row, col)
        else:
            nextPos = self.currentPos

        self.markVisited(nextPos)
        self.freeCellsState = self.getStateFreeCells()
        self.currentPos = nextPos
        if self.currentPos == self.finishPos:
            self.status = WIN
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
        return reward

    def checkValidMove(self, position: tuple) -> bool:
        row, col = position
        if row < 0 or row >= self.mazeRowNb or col < 0 or col >= self.mazeColNb or self.maze[row, col] == WALL:
            return False
        else:
            return True

    def markVisited(self, position: tuple) -> None:
        if self.maze[position] != FINISH:
            self.maze[position] = VISITED

    def getFreeCells(self) -> list:
        freeCells = list()
        for row in range(self.mazeRowNb):
            for col in range(self.mazeColNb):
                if self.maze[row, col] != WALL:
                    freeCells.append((row, col))
        return freeCells

    def getStateFreeCells(self):
        freeCells = list()
        for row in range(self.mazeRowNb):
            for col in range(self.mazeColNb):
                if self.maze[row, col] != WALL:
                    freeCells.append(self.maze[row, col])
        return np.array(freeCells).reshape(1, -1)
