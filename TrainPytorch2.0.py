from py4j.java_gateway import get_field

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from TrainModule import ActionMap
import os
import csv


class TrainPytorch(object):
    # hyper parameters
    EPISODES = 100000  # number of episodes
    EPS_START = 0.9  # e-greedy threshold start value
    EPS_END = 0.05  # e-greedy threshold end value
    EPS_DECAY = 100000  # e-greedy threshold decay
    GAMMA = 0.8  # Q-learning discount factor
    LR = 0.001  # NN optimizer learning rate
    INPUT_SIZE = 141  # input layer size
    HIDDEN_SIZE = 80  # NN hidden layer size
    OUTPUT_SIZE = 40  # output layer size
    BATCH_SIZE = 500  # Q-learning batch size
    Memory_Capacity = 100000  # memoory capacity
    AgentType = 0  # AgentType  0 -> Attack, 1 -> Diffence
    MaxPoint = 120  # max projectile damage (ver 4.10)
    SubPoint = 40  # max damage in usual action (ver 4.10)
    DirName = "attack1"
    SavaInterval = 10  # save per 10 rounds
    RaundFrame = 3600

    # if gpu is to be used
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    Tensor = FloatTensor

    def __init__(self, gateway):
        self.gateway = gateway

    def close(self):
        pass

    def getInformation(self, frameData, nonDelay):
        # Load the frame data every time getInformation gets called
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)
        self.nonDelay = nonDelay
        self.currentFrameNum = nonDelay.getFramesNumber()  # first frame is 14

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
        print(x)
        print(y)
        print(z)

    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        pass

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
        self.player = player  # p1 == True, p2 == False
        self.gameData = gameData
        self.simulator = self.gameData.getSimulator()
        self.isGameJustStarted = True

        self.actionMap = ActionMap()
        self.steps_done = 0  # count of choosing select_action()
        self.currentRoundNum = None  # current round number (the first round num is 0)
        self.R = 0  # total reward in a round
        self.lastHp = 0  # hp (-15 frames)

        # self.modelCount = 0
        # self.randCount = 0

        def makeResultFile():
            if not os.path.exists("./" + self.DirName + "/network"):
                print("make dir")
                os.makedirs("./" + self.DirName + "/network")

            csvList = []
            csvList.append("roundNum")
            csvList.append("R")
            csvList.append("steps")
            csvList.append("myHp")
            csvList.append("oppHp")
            csvList.append("score")
            csvList.append("win")
            f = open("./" + self.DirName + "/resultData.csv", 'a')
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(csvList)
            f.close()

        class ReplayMemory:
            def __init__(self, capacity):
                self.capacity = capacity
                self.memory = []

            def push(self, transition):
                self.memory.append(transition)
                if len(self.memory) > self.capacity:
                    del self.memory[0]

            def sample(self, batch_size):
                return random.sample(self.memory, batch_size)

            def __len__(self):
                return len(self.memory)

        class Network(nn.Module):
            def __init__(self, inputSize, hiddenSize, outputSize):
                nn.Module.__init__(self)
                # self.l1 = nn.Linear(inputSize, hiddenSize)
                # self.l2 = nn.Linear(hiddenSize, outputSize)

                self.l1 = nn.Linear(inputSize, hiddenSize)
                self.l2 = nn.Linear(hiddenSize, hiddenSize)
                self.l3 = nn.Linear(hiddenSize, outputSize)

            def forward(self, x):
                # x = F.relu(self.l1(x))
                # x = self.l2(x)

                x = F.relu(self.l1(x))
                x = F.relu(self.l2(x))
                x = self.l3(x)

                return x

        makeResultFile()
        self.memory = ReplayMemory(self.Memory_Capacity)
        self.model = Network(self.INPUT_SIZE, self.HIDDEN_SIZE, self.OUTPUT_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), self.LR)

        # how to load the net work
        # self.model = TheModelClass(*args, **kwargs)
        # self.model.load_state_dict(torch.load(PATH))

        # self.model = Network(self.INPUT_SIZE, self.HIDDEN_SIZE, self.OUTPUT_SIZE)
        # self.model.load_state_dict(torch.load("./network/dqn_model.pth"))

        return 0

    def input(self):
        return self.inputKey

    def processing(self):
        # First we check whether we are at the end of the round
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            self.isGameJustStarted = True
            return
        if not self.isGameJustStarted:
            # Simulate the delay and look ahead 2 frames. The simulator class exists already in FightingICE
            self.frameData = self.simulator.simulate(self.frameData, self.player, None, None, 17)
        # You can pass actions to the simulator by writing as follows:
        # actions = self.gateway.jvm.java.util.ArrayDeque()
        # actions.add(self.gateway.jvm.enumerate.Action.STAND_A)
        # self.frameData = self.simulator.simulate(self.frameData, self.player, actions, actions, 17)
        else:
            # this else is used only 1 time in first of round
            self.isGameJustStarted = False
            self.currentRoundNum = self.frameData.getRound()
            self.R = 0
            self.isFinishd = 0
        # self.modelCount = 0
        # self.randCount = 0

        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            return
        self.inputKey.empty()
        self.cc.skillCancel()

        # ------------------------------------------------------
        # my code

        def select_action(state):
            sample = random.random()
            decay = (self.EPS_START - self.EPS_END) / self.EPS_DECAY
            eps_threshold = self.EPS_START - (decay * self.steps_done)
            # eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                print("model")
                # self.modelCount += 1
                return self.model(Variable(state, volatile=True).type(self.FloatTensor)).data.max(1)[1].view(1, 1)
            else:
                print("random")
                # self.randCount += 1
                return self.LongTensor([[random.randrange(40)]])

        def getObservation():
            my = self.frameData.getCharacter(self.player)
            opp = self.frameData.getCharacter(not self.player)

            myHp = abs(my.getHp() / 500)
            myEnergy = my.getEnergy() / 300
            myX = ((my.getLeft() + my.getRight()) / 2) / 960
            myY = ((my.getBottom() + my.getTop()) / 2) / 640
            mySpeedX = my.getSpeedX() / 15
            mySpeedY = my.getSpeedY() / 28
            myState = my.getAction().ordinal()

            oppHp = abs(opp.getHp() / 500)
            oppEnergy = opp.getEnergy() / 300
            oppX = ((opp.getLeft() + opp.getRight()) / 2) / 960
            oppY = ((opp.getBottom() + opp.getTop()) / 2) / 640
            oppSpeedX = opp.getSpeedX() / 15
            oppSpeedY = opp.getSpeedY() / 28
            oppState = opp.getAction().ordinal()
            oppRemainingFrame = opp.getRemainingFrame() / 70

            observation = []
            observation.append(myHp)
            observation.append(myEnergy)
            observation.append(myX)
            observation.append(myY)
            if mySpeedX < 0:
                observation.append(0)
            else:
                observation.append(1)
            observation.append(abs(mySpeedX))
            if mySpeedY < 0:
                observation.append(0)
            else:
                observation.append(1)
            observation.append(abs(mySpeedY))
            for i in range(56):
                if i == myState:
                    observation.append(1)
                else:
                    observation.append(0)

            observation.append(oppHp)
            observation.append(oppEnergy)
            observation.append(oppX)
            observation.append(oppY)
            if oppSpeedX < 0:
                observation.append(0)
            else:
                observation.append(1)
            observation.append(abs(oppSpeedX))
            if oppSpeedY < 0:
                observation.append(0)
            else:
                observation.append(1)
            observation.append(abs(oppSpeedY))
            for i in range(56):
                if i == oppState:
                    observation.append(1)
                else:
                    observation.append(0)
            observation.append(oppRemainingFrame)

            myProjectiles = self.frameData.getProjectilesByP1()
            oppProjectiles = self.frameData.getProjectilesByP2()

            if len(myProjectiles) == 2:
                myHitDamage = myProjectiles[0].getHitDamage() / 200.0
                myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                    0].getCurrentHitArea().getRight()) / 2) / 960.0
                myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                    0].getCurrentHitArea().getBottom()) / 2) / 640.0
                observation.append(myHitDamage)
                observation.append(myHitAreaNowX)
                observation.append(myHitAreaNowY)
                myHitDamage = myProjectiles[1].getHitDamage() / 200.0
                myHitAreaNowX = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[
                    1].getCurrentHitArea().getRight()) / 2) / 960.0
                myHitAreaNowY = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[
                    1].getCurrentHitArea().getBottom()) / 2) / 640.0
                observation.append(myHitDamage)
                observation.append(myHitAreaNowX)
                observation.append(myHitAreaNowY)
            elif len(myProjectiles) == 1:
                myHitDamage = myProjectiles[0].getHitDamage() / 200.0
                myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                    0].getCurrentHitArea().getRight()) / 2) / 960.0
                myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                    0].getCurrentHitArea().getBottom()) / 2) / 640.0
                observation.append(myHitDamage)
                observation.append(myHitAreaNowX)
                observation.append(myHitAreaNowY)
                for t in range(3):
                    observation.append(0.0)
            else:
                for t in range(6):
                    observation.append(0.0)

            if len(oppProjectiles) == 2:
                oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
                oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                    0].getCurrentHitArea().getRight()) / 2) / 960.0
                oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                    0].getCurrentHitArea().getBottom()) / 2) / 640.0
                observation.append(oppHitDamage)
                observation.append(oppHitAreaNowX)
                observation.append(oppHitAreaNowY)
                oppHitDamage = oppProjectiles[1].getHitDamage() / 200.0
                oppHitAreaNowX = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[
                    1].getCurrentHitArea().getRight()) / 2) / 960.0
                oppHitAreaNowY = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[
                    1].getCurrentHitArea().getBottom()) / 2) / 640.0
                observation.append(oppHitDamage)
                observation.append(oppHitAreaNowX)
                observation.append(oppHitAreaNowY)
            elif len(oppProjectiles) == 1:
                oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
                oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                    0].getCurrentHitArea().getRight()) / 2) / 960.0
                oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                    0].getCurrentHitArea().getBottom()) / 2) / 640.0
                observation.append(oppHitDamage)
                observation.append(oppHitAreaNowX)
                observation.append(oppHitAreaNowY)
                for t in range(3):
                    observation.append(0.0)
            else:
                for t in range(6):
                    observation.append(0.0)

            # print(len(observation))  #141
            # type(observation) -> list
            return observation

        def makeReward(finishRound):
            if finishRound == 0:
                if self.AgentType == 0:  # AttackType
                    # reward = currentOppHp - lastOppHp
                    reward = abs(self.nonDelay.getCharacter(not self.player).getHp()) - self.lastHp
                    self.R += reward
                    print(reward)
                    return reward

                elif self.AgentType == 1:  # DiffenceType
                    # reward = SubPoint - (currentMyHp - lastMyHp )
                    reward = self.SubPoint - (abs(self.nonDelay.getCharacter(self.player).getHp()) - self.lastHp)
                    self.R += reward
                    print(reward)
                    return reward

            else:
                if abs(self.nonDelay.getCharacter(self.player).getHp()) < abs(
                        self.nonDelay.getCharacter(not self.player).getHp()):
                    self.R += self.MaxPoint
                    self.win = 1
                    return self.MaxPoint
                else:
                    self.win = 0
                    return 0

        def setLastHp():
            if self.AgentType == 0:  # Attack Type
                self.lastHp = abs(self.nonDelay.getCharacter(not self.player).getHp())
            elif self.AgentType == 1:  # Diffence Type
                self.lastHp = abs(self.nonDelay.getCharacter(self.player).getHp())

        def ableAction():
            if self.nonDelay.getCharacter(self.player).isControl() == True and self.isFinishd == 0:
                return True
            else:
                return False

        def playAction(actionNum):
            actionName = self.actionMap.actionMap[actionNum]
            self.cc.commandCall(actionName)

        def pushData(next_state, reward):
            self.memory.push((self.FloatTensor([self.state]), self.action, self.FloatTensor([next_state]),
                              self.FloatTensor([reward])))

        def learn():
            if len(self.memory) < self.BATCH_SIZE:
                return

            # random transition batch is taken from experience replay memory
            # type(transitions) -> class "list"
            # example batch size is 2
            # transitions = [([state],
            # action,
            # [next_state],
            # reward),
            # ([state],
            # action,
            # [next_state],
            # reward)]
            transitions = self.memory.sample(self.BATCH_SIZE)

            # batch_state -> 1*4,1*4
            batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
            # batch_state -> 2*4
            batch_state = Variable(torch.cat(batch_state))

            batch_action = Variable(torch.cat(batch_action))
            batch_reward = Variable(torch.cat(batch_reward))
            batch_next_state = Variable(torch.cat(batch_next_state))

            # current Q values are estimated by NN for all actions
            current_q_values = self.model(batch_state).gather(1, batch_action)
            # expected Q values are estimated from actions which gives maximum Q value
            max_next_q_values = self.model(batch_next_state).detach().max(1)[0]
            expected_q_values = batch_reward + (self.GAMMA * max_next_q_values)

            # loss is measured from error between current and newly expected Q values
            loss = F.smooth_l1_loss(current_q_values, expected_q_values)

            # backpropagation of loss to NN
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def saveResultData():
            score = (self.nonDelay.getCharacter(not self.player).getHp() / (
                        self.nonDelay.getCharacter(not self.player).getHp() + self.nonDelay.getCharacter(
                    self.player).getHp())) * 1000
            csvList = []
            csvList.append(self.currentRoundNum)
            csvList.append(self.R)
            csvList.append(self.steps_done)
            csvList.append(abs(self.nonDelay.getCharacter(self.player).getHp()))
            csvList.append(abs(self.nonDelay.getCharacter(not self.player).getHp()))
            csvList.append(score)
            csvList.append(self.win)
            # csvList.append(self.modelCount)
            # csvList.append(self.randCount)
            f = open("./" + self.DirName + "/resultData.csv", 'a')
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(csvList)
            f.close()

        def saveNetwork():
            if self.currentRoundNum % self.SavaInterval == 0:
                torch.save(self.model.state_dict(),
                           "./" + self.DirName + "/network/" + str(self.currentRoundNum) + ".pth")
                print("save network")

        # this is main process
        if self.currentFrameNum == 14:
            print("first")
            self.state = getObservation()
            setLastHp()
            action = select_action(self.FloatTensor([self.state]))
            self.action = action
            playAction(action[0, 0])
        elif self.currentFrameNum > (self.RaundFrame - 200) and self.isFinishd == 0:
            reward = makeReward(1)
            state = getObservation()
            pushData(state, reward)
            learn()
            saveResultData()
            saveNetwork()
            self.isFinishd = 1

        elif ableAction():
            reward = makeReward(0)
            state = getObservation()
            pushData(state, reward)
            learn()
            setLastHp()
            self.state = state
            action = select_action(self.FloatTensor([state]))
            self.action = action
            playAction(action[0, 0])

    class Java:
        implements = ["aiinterface.AIInterface"]
