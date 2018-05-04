from py4j.java_gateway import get_field
# from RL_brain import BrainDQN
from RL_brain import BrainDQN
from TrainModule import ActionMap
import numpy as np
import os
import csv
# import crash_on_ipy
# import ipdb;ipdb.set_trace()
# this is cnn
actions = 40

class tensorflow_agent(object):
    def __init__(self, gateway):
        self.gateway = gateway
        self.brain = BrainDQN(actions, 141)
        self.actionMap = ActionMap()
        self.R = 0  # total reward in a round
        self.action = 0
        self.MaxPoint = 120  # max projectile damage (ver 4.10)
        self.SubPoint = 0  # max damage in usual action (ver 4.10)
        self.countProcess = 0
        self.frameData = None
        self.nonDelay = None
        self.currentFrameNum = None
        self.inputKey = None
        self.cc = None
        self.player = None
        self.simulator = None
        self.lastHp_opp = None
        self.lastHp_my = None
        self.isGameJustStarted = None
        self.currentRoundNum = None
        self.isFinishd = None
        self.reward = None
        self.frame_per_action = self.brain.frame_per_action
        self.screenData = None
        self.win = None
        self.state = None
        self.width = self.brain.width
        self.height = self.brain.height

    def close(self):
        pass

    def getInformation(self, frameData, nonDelay):
        # Getting the frame data of the current frame
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)
        self.nonDelay = nonDelay
        self.currentFrameNum = nonDelay.getFramesNumber()  # first frame is 14

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
        try:
            score = (self.nonDelay.getCharacter(not self.player).getHp() / (
            self.nonDelay.getCharacter(not self.player).getHp() + self.nonDelay.getCharacter(self.player).getHp())) * 1000
            csvList = []
            csvList.append(self.currentRoundNum)
            csvList.append(self.R)
            csvList.append(self.brain.epsilon)
            csvList.append(abs(self.nonDelay.getCharacter(self.player).getHp()))
            csvList.append(abs(self.nonDelay.getCharacter(not self.player).getHp()))
            csvList.append(score)
            csvList.append(self.win)
            with open("./saved_networks/resultData.csv", 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(csvList)
            # with open('./saved_networks/battleResult.csv', 'a') as file:
            # file.write("The current step is: " + str(self.brain.session.run(self.brain.timeStep)))
            # file.write("  frame number: " + str(z) + "  p1: " + str(x) + "  p2: " + str(y))
            # file.write("\n")
            print(x)
            print(y)
            print(z)
        except Exception as e:
            print(e)

    def makeResultFile(self):
        if not os.path.exists("./saved_networks/"):
            print("Make direction")
            os.makedirs("./saved_networks/")
        if os.path.isfile('./saved_networks/resultData.csv') == False:
            with open('./saved_networks/resultData.csv', 'w') as file:
                file.write('')
        csvList = []
        csvList.append("roundNum")
        csvList.append("R")
        csvList.append("epsilon")
        csvList.append("myHp")
        csvList.append("oppHp")
        csvList.append("score")
        csvList.append("win")
        f = open("./saved_networks/resultData.csv", 'a')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(csvList)
        f.close()

    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        self.screenData = sd

    def getObservation(self):
        imageData = self.screenData.getDisplayByteBufferAsBytes(self.width, self.height, True)
        imageData = [int(i) for i in imageData]
        imageData = np.array(imageData)
        return imageData.reshape((self.width, self.height, 1))

    def initialize(self, gameData, player):
        # Initializing the command center, the simulator and some other things
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()

        self.player = player
        self.simulator = gameData.getSimulator()
        self.makeResultFile()

        return 0

    def input(self):
        # Return the input for the current frame
        return self.inputKey

    def playAction(self):
        state = self.getObservation()
        self.action = np.argmax(self.brain.get_action(state))
        action_name = self.actionMap.actionMap[self.action]
        print("current action is: ", action_name)
        self.cc.commandCall(action_name)

    def makeReward(self, finishRound):
        if finishRound == 0:
            # Defence reward = SubPoint - (currentMyHp - lastMyHp )
            # Attack reward = currentOppHp - lastOppHp
            self.reward = (self.SubPoint - (abs(self.nonDelay.getCharacter(self.player).getHp()) - self.lastHp_my))
            self.reward += 1 * (abs(self.nonDelay.getCharacter(not self.player).getHp()) - self.lastHp_opp)

            self.R += self.reward
            print("The reward is: ", self.reward)
            return self.reward

        else:
            if abs(self.nonDelay.getCharacter(self.player).getHp()) < abs(
                    self.nonDelay.getCharacter(not self.player).getHp()):
                self.R += self.MaxPoint
                self.win = 1
                return self.MaxPoint
            else:
                self.win = 0
                return 0

    def setLastHp(self):
        self.lastHp_opp = abs(self.nonDelay.getCharacter(not self.player).getHp())
        self.lastHp_my = abs(self.nonDelay.getCharacter(self.player).getHp())

    def ableAction(self):
        if self.nonDelay.getCharacter(self.player).isControl() == True and self.isFinishd == 0:
            return True
        else:
            return False

    def processing(self):
        try:
            # First we check whether we are at the end of the round
            self.frame_per_action -= 1
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

            if self.cc.getSkillFlag():
                self.inputKey = self.cc.getSkillKey()
                return
            self.inputKey.empty()
            self.cc.skillCancel()

            # Just spam kick
            # self.cc.commandCall("B")
            if self.currentFrameNum == 14:
                self.state = self.getObservation()
                self.brain.first_store(self.state)
                self.setLastHp()
                self.playAction()

            elif self.currentFrameNum > 3550 and self.isFinishd == 0:
                reward = self.makeReward(1)
                state_ = self.getObservation()
                self.brain.store_transition(self.action, reward, state_)

                self.playAction()
                self.isFinishd = 1

            elif self.ableAction():
                self.brain.learn()
                if self.frame_per_action <= 0:
                    reward = self.makeReward(0)
                    state_ = self.getObservation()
                    self.brain.store_transition(self.action, reward, state_)

                    self.setLastHp()
                    self.playAction()
                    self.frame_per_action = self.brain.frame_per_action
                    print("\n")

            self.countProcess += 1
        except Exception as e:
            print(e)

    class Java:
        implements = ["aiinterface.AIInterface"]
