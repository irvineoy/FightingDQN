import tensorflow as tf
import numpy as np
import random
from collections import deque
import threading
from tensorflow.python import debug as tf_debug


# This is multihead
# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 300.  # timesteps to observe before training
EXPLORE = 20000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0.9  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100
SAVE_AFTER_STEP = 10000
REWARD_MAX = 40.0
LR = 1e-6

try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply


class BrainDQN:
    def __init__(self, actions):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = tf.Variable(0, trainable=False)
        self.observe_count = 0
        self.costValue = []
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput, self.QValue_attack, self.QValue_defence, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2,\
            self.W_fc3_attack, self.b_fc3_attack, self.W_fc3_defence, self.b_fc3_defence = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValue_attackT, self.QValue_defenceT, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T,\
            self.W_fc3_attackT, self.b_fc3_attackT, self.W_fc3_defenceT, self.b_fc3_defenceT = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2),
                                            self.W_fc3_attackT.assign(self.W_fc3_attack),
                                            self.b_fc3_attackT.assign(self.b_fc3_attack),
                                            self.W_fc3_defenceT.assign(self.W_fc3_defence),
                                            self.b_fc3_defenceT.assign(self.b_fc3_defence)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=4)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        self.copyTargetQNetwork()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def createQNetwork(self):
        W_fc1 = self.weight_variable([141*4, 1024], "fc1")
        b_fc1 = self.bias_variable([1024], "fc1")

        W_fc2 = self.weight_variable([1024, 1024], "fc2")
        b_fc2 = self.bias_variable([1024], "fc2")

        W_fc3_attack = self.weight_variable([1024, self.actions], "fc3_attack")
        b_fc3_attack = self.bias_variable([self.actions], "fc3_attack")

        W_fc3_defence = self.weight_variable([1024, self.actions], "fc3_defence")
        b_fc3_defence = self.bias_variable([self.actions], "fc3_defence")

        # input layer
        stateInput = tf.placeholder("float", [None, 141, 4])

        h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(stateInput, [-1, 141*4]), W_fc1), b_fc1))
        h_fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2))

        # Q Value layer
        QValue_attack = tf.matmul(h_fc2, W_fc3_attack) + b_fc3_attack
        QValue_defence = tf.matmul(h_fc2, W_fc3_defence) + b_fc3_defence

        # stateInput = tf.placeholder("float", [None, 141])
        # h_fc1 = tf.layers.dense(inputs=stateInput, units=80, activation=tf.nn.relu)

        return stateInput, QValue_attack, QValue_defence, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3_attack, b_fc3_attack, \
            W_fc3_defence, b_fc3_defence

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])

        Q_Action_attack = tf.reduce_sum(tf.mul(self.QValue_attack, self.actionInput), reduction_indices=1)
        self.cost_attack = tf.reduce_mean(tf.square(self.yInput - Q_Action_attack))
        self.trainStep_attack = tf.train.AdamOptimizer(LR).minimize(self.cost_attack, global_step=self.timeStep)

        Q_Action_defence = tf.reduce_sum(tf.mul(self.QValue_defence, self.actionInput), reduction_indices=1)
        self.cost_defence = tf.reduce_mean(tf.square(self.yInput - Q_Action_defence))
        self.trainStep_defence = tf.train.AdamOptimizer(LR).minimize(self.cost_defence, global_step=self.timeStep)

    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [list(data[3]) for data in minibatch]

        self.costValue = []
        # Step 2: calculate y attack
        y_batch = []
        QValue_batch = self.session.run(self.QValue_attackT, feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            # terminal = minibatch[i][4]
            # if terminal:
            #     y_batch.append(reward_batch[i][1])
            # else:
            y_batch.append(reward_batch[i][1] + GAMMA * np.max(QValue_batch[i]))

        cost_attack = self.session.run([self.cost_attack, self.trainStep_attack], feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })
        self.costValue.append(cost_attack[0])
        # print("The attack cost is: ", cost_attack[0])

        # Step 2: calculate y defence
        y_batch = []
        QValue_batch = self.session.run(self.QValue_defenceT, feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            # terminal = minibatch[i][4]
            # if terminal:
            #     y_batch.append(reward_batch[i][0])
            # else:
            y_batch.append(reward_batch[i][0] + GAMMA * np.max(QValue_batch[i]))

        cost_defence = self.session.run([self.cost_defence, self.trainStep_defence], feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })
        self.costValue.append(cost_defence[0])
        # print("The defence cost is: ", cost_defence[0])

        # save network every 100000 iteration
        if self.session.run(self.timeStep) % SAVE_AFTER_STEP == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-ddqn', global_step=self.timeStep)

        if self.session.run(self.timeStep) % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, nextObservation, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = np.append(self.currentState[:, 1:], np.reshape(nextObservation, (141, 1)), axis=1)
        reward_normalize = [i/REWARD_MAX for i in reward]
        reward_normalize = []
        reward_normalize.append(reward[0] / REWARD_MAX)
        reward_normalize.append(reward[1] / REWARD_MAX)
        self.replayMemory.append((self.currentState, action, reward_normalize, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.observe_count > OBSERVE:
            # Train the network
            self.trainQNetwork()
        # print info
        if self.observe_count < OBSERVE:
            state = "observe"
            self.observe_count += 1
        elif self.observe_count == OBSERVE:
            t = threading.Thread(target=self.trainAllTheTime, args=())
            t.start()
            state = "begin the threading"
            self.observe_count += 1
        elif self.session.run(self.timeStep) <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.session.run(self.timeStep), "/ STATE", state, \
              "/ EPSILON", self.epsilon)

        self.currentState = newState
        # self.timeStep += 1

    def getAction(self):
        # QValue = self.QValue.eval(feed_dict={self.stateInput: self.currentState})[0]
        QValue_attack = self.session.run(self.QValue_attack, feed_dict={self.stateInput: [self.currentState]})[0]
        QValue_defence = self.session.run(self.QValue_defence, feed_dict={self.stateInput: [self.currentState]})[0]
        QValue = QValue_attack + QValue_defence
        action = np.zeros(self.actions)
        action_index = 0
        if self.session.run(self.timeStep) % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.session.run(self.timeStep) > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def setInitState(self, observation):
        # init_g = tf.global_variables_initializer()
        # init_l = tf.local_variables_initializer()
        # with tf.Session() as sess:
        #     with tf_debug.LocalCLIDebugWrapperSession(sess) as sess:
        #         sess.run(init_g)
        #         sess.run(init_l)
        self.currentState = np.stack((observation, observation, observation, observation), axis=1)

    def weight_variable(self, shape, name):
        with tf.variable_scope(name):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)

    def bias_variable(self, shape, name):
        with tf.variable_scope(name):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial)

    def trainAllTheTime(self):
        while 1:
            self.trainQNetwork()
            # sleep(0.1)