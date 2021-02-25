from PIL import Image
import numpy as np


class Maze():
    def __init__(self, mazeFilePath) -> None:
        self.filePath = mazeFilePath
        self.loadMaze()

    def loadMaze(self):
        # load file and convert it to bilevel
        img = Image.open(self.filePath)
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

        self.startPos = (0, np.where(mazeArray[0] == 1)[0][0])
        self.finishPos = (mazeArray.shape[0]-1, np.where(mazeArray[mazeArray.shape[0]-1] == 1)[0][0])
        mazeArray[self.finishPos] = 2
        self.maze = mazeArray.astype('int32')

    def move(self, currentPos, direction):
        # directions: 0.UP 1.DOWN 2.LEFT 3.RIGHT
        row = currentPos[0]
        col = currentPos[1]

        if direction == 0:
            row -= 1
        if direction == 1:
            row += 1
        if direction == 2:
            col -= 1
        if direction == 3:
            col += 1

        nextPos = (row, col)
        if row < 0 or row >= self.mazeRowNb or col < 0 or col >= self.mazeColNb:
            reward = -0.9
            nextPos = currentPos
        elif self.maze[row, col] == 0:
            reward = -0.75
            nextPos = currentPos
        elif self.maze[row, col] == 1 or self.maze[row, col] == 3:
            reward = -0.03
        elif self.maze[row, col] == 2:
            reward = 1

        return nextPos, reward

    def possibleDirections(self, currentPos, visited):
        possibleDirs = []
        for dir in range(4):
            nextPos = self.move(currentPos, dir)[0]
            if nextPos != currentPos and nextPos not in visited:
                possibleDirs.append(dir)
        possibleDirs = [0,1,2,3]
        return possibleDirs
