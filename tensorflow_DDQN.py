import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.python import debug as tf_debug
import threading
from time import sleep


# This is master
# Hyper Parameters:

FRAME_PER_ACTION = 5
GAMMA = 0.90  # decay rate of past observations
OBSERVE = 500.  # timesteps to observe before training
EXPLORE = 10000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.1  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0.9  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 300
SAVE_AFTER_STEP = 10000
REWARD_MAX = 40.0
LR = 1e-3

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
        self.frame_per_action = FRAME_PER_ACTION
        self.timeStep = tf.Variable(0, trainable=False)
        self.observe_count = 0
        self.costValue = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.frame_per_action = FRAME_PER_ACTION

        eval_names = ['eval_network', tf.GraphKeys.GLOBAL_VARIABLES]
        target_names = ['target_network', tf.GraphKeys.GLOBAL_VARIABLES]
        self.stateInput, self.QValue, self.QValue_a, self.QValue_d = self.createQNetwork(eval_names)
        self.stateInputT, self.QValueT, self.QValue_aT, self.QValue_dT = self.createQNetwork(target_names)
        e_params = tf.get_collection('eval_network')
        t_params = tf.get_collection('target_network')
        self.copyTargetQNetworkOperation = [tf.assign(e, t) for e, t in zip(e_params, t_params)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=1)
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


    def createQNetwork(self, c_name):
        W_fc1 = self.weight_variable([141, 80], "fc1", c_name)
        b_fc1 = self.bias_variable([80], "fc1", c_name)

        W_fc2 = self.weight_variable([80, 80], "fc2", c_name)
        b_fc2 = self.bias_variable([80], "fc2", c_name)

        W_fc3_a = self.weight_variable([80, self.actions], "fc3_attack", c_name)
        b_fc3_a = self.bias_variable([self.actions], "fc3_attack", c_name)

        W_fc3_d = self.weight_variable([80, self.actions], "fc3_defence", c_name)
        b_fc3_d = self.bias_variable([self.actions], "fc3_defence", c_name)

        # input layer
        stateInput = tf.placeholder("float", [None, 141])

        h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(stateInput, W_fc1), b_fc1))
        h_fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2))

        # Q Value layer
        QValue_a = tf.matmul(h_fc2, W_fc3_a) + b_fc3_a
        QValue_d = tf.matmul(h_fc2, W_fc3_d) + b_fc3_d
        QValue = QValue_a + QValue_d

        # h_fc1 = tf.layers.dense(inputs=stateInput, units=80, activation=tf.nn.relu)

        return stateInput, QValue, QValue_a, QValue_d

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput_a = tf.placeholder("float", [None])
        self.yInput_d = tf.placeholder("float", [None])
        Q_Action_a = tf.reduce_sum(tf.mul(self.QValue_a, self.actionInput), reduction_indices=1)
        Q_Action_d = tf.reduce_sum(tf.mul(self.QValue_d, self.actionInput), reduction_indices=1)
        cost_a = tf.reduce_mean(tf.square(self.yInput_a - Q_Action_a))
        cost_d = tf.reduce_mean(tf.square(self.yInput_d - Q_Action_d))
        self.cost = cost_a + cost_d
        self.trainStep = tf.train.AdamOptimizer(LR).minimize(self.cost, global_step=self.timeStep)

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [list(data[3]) for data in minibatch]

        # Step 2: calculate y
        y_batch_a = []
        QValue_batch_a = self.session.run(self.QValue_aT, feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            y_batch_a.append(reward_batch[i][1] + GAMMA * np.max(QValue_batch_a[i]))

        y_batch_d = []
        QValue_batch_d = self.session.run(self.QValue_dT, feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            y_batch_d.append(reward_batch[i][0] + GAMMA * np.max(QValue_batch_d[i]))

        costValue = self.session.run([self.cost, self.trainStep], feed_dict={
            self.yInput_a: y_batch_a,
            self.yInput_d: y_batch_d,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })
        self.costValue = costValue[0]
        # print("The cost is: ", cost[0])

        # save network every 100000 iteration
        if self.session.run(self.timeStep) % SAVE_AFTER_STEP == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-ddqn', global_step=self.timeStep)

        if self.session.run(self.timeStep) % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, nextObservation, action, reward):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        # newState = np.append(self.currentState[:, 1:], np.reshape(nextObservation, (141, 1)), axis=1)
        newState = np.array(nextObservation)
        reward_normalize = [i / REWARD_MAX for i in reward]
        self.replayMemory.append((self.currentState, action, reward_normalize, newState))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.observe_count > OBSERVE:
                pass
            # Train the network
            # self.trainQNetwork()
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
              "/ EPSILON", round(self.epsilon, 6))
        print("The cost is: ", self.costValue)

        self.currentState = newState
        # self.timeStep += 1

    def getAction(self, observation):
        # QValue = self.QValue.eval(feed_dict={self.stateInput: self.currentState})[0]
        QValue = self.session.run(self.QValue, feed_dict={self.stateInput: [observation]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.actions)
            action[action_index] = 1
        else:
            action_index = np.argmax(QValue)
            action[action_index] = 1

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.session.run(self.timeStep) > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action

    def setInitState(self, observation):
        self.currentState = np.array(observation)

    def weight_variable(self, shape, name, collection):
        with tf.variable_scope(name):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial, collections=collection)

    def bias_variable(self, shape, name, collection):
        with tf.variable_scope(name):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial, collections=collection)

    def trainAllTheTime(self):
        while 1:
            self.trainQNetwork()
