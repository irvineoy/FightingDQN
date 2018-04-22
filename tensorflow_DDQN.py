import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.python import debug as tf_debug
import threading

# this is cnn
# Hyper Parameters:
FRAME_PER_ACTION = 10
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 5000.  # timesteps to observe before training
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
        self.frame_per_action = FRAME_PER_ACTION
        self.timeStep = tf.Variable(0, trainable=False)
        self.observe_count = 0
        self.costValue = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_fc1, self.b_fc1, \
            self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3 = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_fc1T, \
            self.b_fc1T, self.W_fc2T, self.b_fc2T, self.W_fc3T, self.b_fc3T = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2),
                                            self.W_fc3T.assign(self.W_fc3), self.b_fc3T.assign(self.b_fc3),
                                            self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2)]

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
        W_conv1 = self.weight_variable([8, 8, 4, 16], "conv1")
        b_conv1 = self.bias_variable([16], "conv1")

        W_conv2 = self.weight_variable([4, 4, 16, 32], "conv2")
        b_conv2 = self.bias_variable([32], "conv2")

        # W_conv3 = self.weight_variable([3, 3, 64, 64], "conv3")
        # b_conv3 = self.bias_variable([64], "conv3")

        W_fc1 = self.weight_variable([1120, 512], "fc1")
        b_fc1 = self.bias_variable([512], "fc1")

        W_fc2 = self.weight_variable([512, 256], "fc2")
        b_fc2 = self.bias_variable([256], "fc2")

        W_fc3 = self.weight_variable([256, self.actions], "fc3")
        b_fc3 = self.bias_variable([self.actions], "fc3")

        # input layer
        stateInput = tf.placeholder("float", [None, 14, 10, 4])

        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 2) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 1) + b_conv2)
        h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(h_conv2, [-1, 1120]), W_fc1), b_fc1))
        h_fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2))

        # Q Value layer
        QValue = tf.matmul(h_fc2, W_fc3) + b_fc3

        # stateInput = tf.placeholder("float", [None, 141])
        # h_fc1 = tf.layers.dense(inputs=stateInput, units=80, activation=tf.nn.relu)

        return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(LR).minimize(self.cost, global_step=self.timeStep)

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [list(data[3]) for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.session.run(self.QValueT, feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            # terminal = minibatch[i][4]
            # if terminal:
            #     y_batch.append(reward_batch[i])
            # else:
            y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        costValue = self.session.run([self.cost, self.trainStep], feed_dict={
            self.yInput: y_batch,
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

    def setPerception(self, nextObservation, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = np.append(self.currentState[:, :, 1:], np.reshape(nextObservation, (14, 10, 1)), axis=2)
        reward_normalize = reward / REWARD_MAX
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
        print("The cost is: ", self.costValue)

        self.currentState = newState
        # self.timeStep += 1

    def getAction(self):
        # QValue = self.QValue.eval(feed_dict={self.stateInput: self.currentState})[0]
        QValue = self.session.run(self.QValue, feed_dict={self.stateInput: [self.currentState]})[0]
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
        # init_g = tf.global_variables_initializer()
        # init_l = tf.local_variables_initializer()
        # with tf.Session() as sess:
        #     with tf_debug.LocalCLIDebugWrapperSession(sess) as sess:
        #         sess.run(init_g)
        #         sess.run(init_l)
        observation = np.reshape(observation, (14, 10))
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)
        print(self.currentState.shape)

    def weight_variable(self, shape, name):
        with tf.variable_scope(name):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)

    def bias_variable(self, shape, name):
        with tf.variable_scope(name):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def trainAllTheTime(self):
        while 1:
            self.trainQNetwork()
