from py4j.java_gateway import get_field
from tensorflow_DDQN import BrainDQN
from TrainModule import ActionMap
import numpy as np
import os
# import crash_on_ipy
# import ipdb;ipdb.set_trace()
actions = 40


class tensorflow_agent(object):
    def __init__(self, gateway):
        self.gateway = gateway
        self.brain = BrainDQN(actions)
        self.actionMap = ActionMap()
        self.R = 0  # total reward in a round
        self.action = 0
        self.MaxPoint = 120  # max projectile damage (ver 4.10)
        self.SubPoint = 10  # max damage in usual action (ver 4.10)
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
        if os.path.isfile('./saved_networks/battleResult.txt') == False:
            with open('./saved_networks/battleResult.txt', 'w') as file:
                file.write('')
        with open('./saved_networks/battleResult.txt', 'a') as file:
            file.write("The current step is: " + str(self.brain.session.run(self.brain.timeStep)))
            file.write("  frame number: " + str(z) + "  p1: " + str(x) + "  p2: " + str(y))
            file.write("\n")
        print(x)
        print(y)
        print(z)

    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        pass

    def initialize(self, gameData, player):
        # Initializing the command center, the simulator and some other things
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()

        self.player = player
        self.simulator = gameData.getSimulator()

        return 0

    def input(self):
        # Return the input for the current frame
        return self.inputKey

    def playAction(self):
        self.action = self.brain.getAction()
        action_name = self.actionMap.actionMap[np.argmax(self.action)]
        print("current action is: ", action_name)
        self.cc.commandCall(action_name)


    def getObservation(self):
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
        return list(map(lambda x: float(x), observation))

    def makeReward(self, finishRound):
        if finishRound == 0:
            # Defence reward = SubPoint - (currentMyHp - lastMyHp )
            # Attack reward = currentOppHp - lastOppHp
            self.reward = []  # 0:defence 1:attack
            self.reward.append(self.SubPoint - (abs(self.nonDelay.getCharacter(self.player).getHp()) - self.lastHp_my))
            self.reward.append(abs(self.nonDelay.getCharacter(not self.player).getHp()) - self.lastHp_opp)

            self.R = self.R + self.reward[0] + self.reward[1]
            print("The reward is: ", self.reward)
            return self.reward

        else:
            if abs(self.nonDelay.getCharacter(self.player).getHp()) < abs(
                    self.nonDelay.getCharacter(not self.player).getHp()):
                self.R += self.MaxPoint
                self.win = 1
                return [self.MaxPoint, self.MaxPoint]
            else:
                self.win = 0
                return [0, 0]

    def setLastHp(self):
        self.lastHp_opp = abs(self.nonDelay.getCharacter(not self.player).getHp())
        self.lastHp_my = abs(self.nonDelay.getCharacter(self.player).getHp())

    def ableAction(self):
        if self.nonDelay.getCharacter(self.player).isControl() == True and self.isFinishd == 0:
            return True
        else:
            return False

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

        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            return
        self.inputKey.empty()
        self.cc.skillCancel()

        # Just spam kick
        # self.cc.commandCall("B")
        if self.currentFrameNum == 14:
            state = self.getObservation()
            self.brain.setInitState(tuple(state))
            self.setLastHp()
            self.playAction()

        elif self.currentFrameNum > 3570 and self.isFinishd == 0:
            reward = self.makeReward(1)
            state = self.getObservation()
            self.playAction()
            self.brain.setPerception(state, self.action, reward, True)
            self.isFinishd = 1

        elif self.ableAction():
            reward = self.makeReward(0)
            state = self.getObservation()
            self.setLastHp()
            self.playAction()
            print("\n")
            self.brain.setPerception(state, self.action, reward, False)

        # print("The countProcess: ", self.countProcess)
        self.countProcess += 1

        # nextObservation = self.getObservation()
        # reward = self.makeReward(self.isGameJustStarted)
        # print(reward)
        # self.brain.setPerception(nextObservation, self.action, reward, self.isGameJustStarted)

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
