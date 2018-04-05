import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.python import debug as tf_debug


# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 20.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 10  # size of minibatch
UPDATE_TIME = 100

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
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput, self.QValue, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3 = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T, self.W_fc3T, self.b_fc3T = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2),
                                            self.W_fc3T.assign(self.W_fc3), self.b_fc3T.assign(self.b_fc3)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
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
        W_fc1 = self.weight_variable([141*4, 80], "fc1")
        b_fc1 = self.bias_variable([80], "fc1")

        W_fc2 = self.weight_variable([80, 80], "fc2")
        b_fc2 = self.bias_variable([80], "fc2")

        W_fc3 = self.weight_variable([80, self.actions], "fc3")
        b_fc3 = self.bias_variable([self.actions], "fc3")

        # input layer
        stateInput = tf.placeholder("float", [None, 141, 4])

        h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(stateInput, [-1, 141*4]), W_fc1), b_fc1))
        h_fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2))

        # Q Value layer
        QValue = tf.matmul(h_fc2, W_fc3) + b_fc3

        # stateInput = tf.placeholder("float", [None, 141])
        # h_fc1 = tf.layers.dense(inputs=stateInput, units=80, activation=tf.nn.relu)

        return stateInput, QValue, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

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
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.session.run(self.trainStep, feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, nextObservation, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = np.append(self.currentState[:, 1:], np.reshape(nextObservation, (141, 1)), axis=1)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()
        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ STATE", state, \
              "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        # QValue = self.QValue.eval(feed_dict={self.stateInput: self.currentState})[0]
        QValue = self.session.run(self.QValue, feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
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


def playFlappyBird():
    # Step 1: init BrainDQN
    actions = 40
    brain = BrainDQN(actions)
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1, 0])  # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    # Step 3.2: run the game
    while 1 != 0:
        action = brain.getAction()
        nextObservation, reward, terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation, action, reward, terminal)