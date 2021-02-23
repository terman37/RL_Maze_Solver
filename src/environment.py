import pygame as pg
import numpy as np
from PIL import Image


class Environment():
    def __init__(self, filePath: str) -> None:

        self.filePath = filePath

        self.loadMaze()
        self.initDisplay()

    def loadMaze(self):
        # load file and convert it to bilevel
        img = Image.open(self.filePath)
        img2lvl = img.convert('1')

        # convert to simplified array
        # original image walls are 2px thick and cells are 14x14px
        npImg = np.asarray(img2lvl)
        self.originalImgSize = (npImg.shape[1], npImg.shape[0])

        mazeRowNb = int((npImg.shape[0]-2)/16)*2+1
        mazeColNb = int((npImg.shape[1]-2)/16)*2+1

        mazeArray = np.empty((mazeRowNb, mazeColNb))
        for row in range(0, mazeRowNb):
            for col in range(0, mazeColNb):
                mazeArray[row, col] = npImg[row * 8, col * 8]
        self.maze = mazeArray.astype('int32')

    def initDisplay(self):
        # reshape factor to avoid app bigger than screen
        maxWidth = 900
        maxHeight = 900

        rFactor = 1
        if self.originalImgSize[0] > maxWidth or self.originalImgSize[1] > maxHeight:
            rFactor = round(
                min(maxWidth/self.originalImgSize[0], maxHeight/self.originalImgSize[1]), 2)

        # init app screen
        self.screen = pg.display.set_mode((int(rFactor * self.originalImgSize[0]*2+150),
                                           int(rFactor * self.originalImgSize[1]+100)))
        self.screen.fill([127, 127, 127])
        pg.display.set_caption('Maze Solver using QLearning')

        # original maze png
        self.image = pg.image.load(self.filePath)
        self.image = pg.transform.scale(self.image, tuple(
            int(i * rFactor) for i in self.originalImgSize))
        self.screen.blit(self.image, (50, 50))

        # simplified maze (rescaled for display)
        colors = np.array([[0, 0, 0], [255, 255, 255]])
        surface = pg.surfarray.make_surface(colors[np.transpose(self.maze)])
        surface = pg.transform.scale(surface, tuple(
            int(i * rFactor) for i in self.originalImgSize))
        self.screen.blit(
            surface, (int(rFactor * self.originalImgSize[0]+100), 50))

        # refresh
        pg.display.flip()


if __name__ == "__main__":

    mazepath = './maze_pictures/10x10maze.png'
    env = Environment(mazepath)

    clock = pg.time.Clock()
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        clock.tick(10)
