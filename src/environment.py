import pygame as pg
import numpy as np
from PIL import Image


class Environment():
    def __init__(self, filePath: str) -> None:
        pg.init()
        self.filePath = filePath

        self.loadMaze()
        self.initDisplay()
        self.initRewardBoard()

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
        self.finishPos = (
            mazeArray.shape[0]-1, np.where(mazeArray[mazeArray.shape[0]-1] == 1)[0][0])
        mazeArray[self.finishPos] = 2
        self.maze = mazeArray.astype('int32')

    def initDisplay(self):
        # reshape factor to avoid app bigger than screen
        maxWidth = 900
        maxHeight = 900

        self.rFactor = 1
        if self.originalImgSize[0] > maxWidth or self.originalImgSize[1] > maxHeight:
            self.rFactor = round(
                min(maxWidth/self.originalImgSize[0], maxHeight/self.originalImgSize[1]), 2)

        # init app screen
        self.screen = pg.display.set_mode((int(self.rFactor * self.originalImgSize[0]+100),
                                           int(self.rFactor * self.originalImgSize[1]+100)))
        self.screen.fill([127, 127, 127])
        pg.display.set_caption('Maze Solver using QLearning')

        # refresh
        pg.display.flip()

    def displayMaze(self):
        # simplified maze (rescaled for display)
        colors = np.array(
            [[0, 0, 0], [255, 255, 255], [0, 255, 0], [127, 255, 255]])
        surface = pg.surfarray.make_surface(colors[np.transpose(self.maze)])
        surface = pg.transform.scale(surface, tuple(
            int(i * self.rFactor) for i in self.originalImgSize))
        self.screen.blit(surface, (50, 50))

        # refresh
        pg.display.flip()

    def displayText(self, text):
        pg.draw.rect(self.screen, (127, 127, 127), (50, 10, 200, 30))
        myfont = pg.font.Font(pg.font.get_default_font(), 20)
        text_surface = myfont.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (55, 15))

        # refresh
        pg.display.flip()

    def initRewardBoard(self):

        # Rewards:
        # wall = move not allowed =0
        # living penalty = -3
        # finish reward = 1000
        # no move living penalty = -100

        self.rewards = [0, -3, 1000, -100]
        self.rewardBoard = np.zeros(
            (self.mazeRowNb*self.mazeColNb, self.mazeRowNb*self.mazeColNb))

        for row in range(self.mazeRowNb):
            for col in range(self.mazeColNb):
                fromPos = self.mazeColNb * row + col
                # if not a wall
                if self.maze[row, col] != 0:
                    # move left
                    if col > 0:
                        toPos = (self.mazeColNb) * row + (col-1)
                        self.rewardBoard[fromPos,
                                         toPos] = self.rewards[self.maze[row, col-1]]
                    # move right
                    if col < self.mazeColNb-1:
                        toPos = (self.mazeColNb) * row + (col+1)
                        self.rewardBoard[fromPos,
                                         toPos] = self.rewards[self.maze[row, col+1]]
                    # move up
                    if row > 0:
                        toPos = self.mazeColNb * (row-1) + col
                        self.rewardBoard[fromPos,
                                         toPos] = self.rewards[self.maze[row-1, col]]
                    # move down
                    if row < self.mazeRowNb-1:
                        toPos = self.mazeColNb * (row+1) + col
                        self.rewardBoard[fromPos,
                                         toPos] = self.rewards[self.maze[row+1, col]]
                    # no move
                        # self.rewardBoard[fromPos, fromPos] = self.rewards[3]

    def moveInMaze(self, nextPos):
        self.maze[nextPos // self.mazeColNb, nextPos % self.mazeColNb] = 3
        self.displayMaze()
        pg.time.wait(50)