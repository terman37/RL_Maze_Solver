# Maze Solver using Reinforcement Learning

<img src="pictures/result_maze2_qtable.gif" alt="result_maze2_qtable" style="zoom:50%;" />

## Context

The idea is to use RL techniques to solve mazes.

## Generate maze

Mazes can be generated using [mazegenerator.net](http://www.mazegenerator.net/) 

Simply select width and height, click generate and  download maze as png.

<img src="pictures/mazegenerator.png" style="zoom: 67%;" />

Restrictions: 

- for now, only rectangular orthogonal mazes are supported. I plan to be able to solve any type of maze...

## QLearning

### Intro

First and most simple implementation of QLearning is using what is called QTable. The main idea behind it is to calculate for each cell the corresponding QValue for each possible action.

There is plenty of very good articles on the web explaining in details QTable learning, so i will only concentrate on the basics and implementation:

All is based on the Bellman's equation and Markov Decision Processes (MDP) giving the value of a state:

<img src="pictures/bellman.png" alt="bellman" style="zoom: 33%;" />

and derived from it the QValues (value of an action) update equation

![](pictures/Qlearning.png)

### Pseudo code

```
# Training
initialize the QTable with reward values

repeat n times:
	- select a random allowed position on the maze
	- select a random possible direction (UP:DOWN:LEFT:RIGHT)
	- get the reward for this direction from this position
	- update the Qvalue for this direction from this position using above equation

# Playing
start from StartPosition

repeat until reaching the end:
	- Find the direction correspondinng to the max Qvalue at CurrentPosition
	- Move from CurrentPosition using direction found
	- update CurrentPosittion
```

### Results and Limitations

If repeating enough times during the training phase, algorithm works quite well.

I've been able to solve 20x20 maze with 2 to 3 seconds training.

To run the code

```
python qTablelearning.py --mazefilepath <PATH TO MAZE PNG>
```



<img src="pictures/result_maze1_qtable.gif" style="zoom: 80%;" /> <img src="pictures/result_maze2_qtable.gif" style="zoom: 80%;" />



But training time is directly related to nb of cells (states) and increase exponentially.

#### Next step will be to train Neural Networks (DQN) to do the same and compare performances.