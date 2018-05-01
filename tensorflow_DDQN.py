import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.python import debug as tf_debug
import threading

# this is cnn
# Hyper Parameters:
FRAME_PER_ACTION = 5
GAMMA = 0.9  # decay rate of past observations
OBSERVE = 50.  # timesteps to observe before training
EXPLORE = 10000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.1  # 0.001 # final value of epsilon
INITIAL_EPSILON = 1.0  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 300
SAVE_AFTER_STEP = 10000
REWARD_MAX = 40.0
WIDTH = 96 * 4
HEIGHT = 64 * 4
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
        self.width = WIDTH
        self.height = HEIGHT
        self.timeStep = tf.Variable(0, trainable=False)
        self.observe_count = 0
        self.costValue = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network

        eval_names = ['eval_network', tf.GraphKeys.GLOBAL_VARIABLES]
        target_names = ['target_network', tf.GraphKeys.GLOBAL_VARIABLES]

        self.stateInput, self.QValue = self.createQNetwork(eval_names)
        self.stateInputT, self.QValueT = self.createQNetwork(target_names)
        
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
        W_conv1 = self.weight_variable([8, 8, 4, 16], "conv1", c_name)
        b_conv1 = self.bias_variable([16], "conv1", c_name)

        W_conv2 = self.weight_variable([4, 4, 16, 32], "conv2", c_name)
        b_conv2 = self.bias_variable([32], "conv2", c_name)

        W_conv3 = self.weight_variable([3, 3, 32, 64], "conv3", c_name)
        b_conv3 = self.bias_variable([64], "conv3", c_name)

        W_fc1 = self.weight_variable([1536, 512], "fc1", c_name)
        b_fc1 = self.bias_variable([512], "fc1", c_name)

        W_fc2 = self.weight_variable([512, 256], "fc2", c_name)
        b_fc2 = self.bias_variable([256], "fc2", c_name)

        W_fc3 = self.weight_variable([256, self.actions], "fc3", c_name)
        b_fc3 = self.bias_variable([self.actions], "fc3", c_name)

        # input layer
        stateInput = tf.placeholder("float", [None, self.width, self.height, 4])

        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 2) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, 2) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)

        h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(h_pool3, [-1, 1536]), W_fc1), b_fc1))
        h_fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2))

        # Q Value layer
        QValue = tf.matmul(h_fc2, W_fc3) + b_fc3

        # stateInput = tf.placeholder("float", [None, 141])
        # h_fc1 = tf.layers.dense(inputs=stateInput, units=80, activation=tf.nn.relu)

        return stateInput, QValue

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
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        reward_normalize = reward / REWARD_MAX
        self.replayMemory.append((self.currentState, action, reward_normalize, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.observe_count > OBSERVE:
            # Train the network
            # self.trainQNetwork()
            pass
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
        # observation = np.reshape(observation, (14, 10))
        observation = observation.reshape((self.width, self.height))
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)
        # self.currentState = observation

    def weight_variable(self, shape, name, c_name):
        with tf.variable_scope(name):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial, collections=c_name)

    def bias_variable(self, shape, name, c_name):
        with tf.variable_scope(name):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial, collections=c_name)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = "SAME")

    def trainAllTheTime(self):
        while 1:
            self.trainQNetwork()
