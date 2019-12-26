# Applying Hybrid Reward Architecture to a Fighting Game AI
### Introduction
This project is the implementation of the model purposed by the paper published at the IEEE CIG conference (IEEE Conference on Computational Intelligence and Games, Aug 2018).

Paper Link: https://ieeexplore.ieee.org/abstract/document/8490437

### Abstract
In this paper, we propose a method for implementing a competent fighting game AI using Hybrid Reward Architecture (HRA). In 2017, an AI using HRA developed by Seijen et al. achieved a perfect score of 999,990 in Ms. Pac-Man. HRA decomposes a reward function into multiple components and learns a separate value function for each component in the reward function. Due to reward decomposition, an optimal value function can be learned in the domain of Ms. Pac-Man. However, the number of actions in Ms. Pac-Man is only limited to four (Up, Down, Left, and Right), and till now whether HRA is also effective in other games with a larger number of actions is unclear. In this paper, we apply HRA and verify its effectiveness in a fighting game. For performance evaluation, we use FightingICE that has 40 actions and has been used as the game platform in the Fighting Game AI Competition at CIG since 2014. Our experimental results show that the proposed HRA AI, a new sample AI for the competition, is superior to non-HRA deep learning AIs and is competitive against other entries of the 2017 competition.

### The FighingICE game
http://www.ice.ci.ritsumei.ac.jp/~ftgaic/index-1.html

Fighting game is a very challenging and entertaining game genre that requires the player to decide an action to perform among many actions (56 actions in FightingICE) within a short response time (16.67 ms in this game). Our research questions are whether or not it is possible to realize general fighting game AIs and if so how to realize them. By general fighting game AIs, we mean those that are strong against any opponents -- AIs or human players -- at any play modes using any character data. Top AIs in our the most recent competition have shown that the answer to the first question might be "yes", but it might take some time in order to answer the second question. Please join us in a journey to seek those answers!


### Contact
Ouyang （irvineoy@163.com）
