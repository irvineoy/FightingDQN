"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import random

np.random.seed(1)
tf.set_random_seed(1)
FRAME_PER_ACTION = 5
REWARD_MAX = 40.
SAVE_AFTER_STEP = 10000
WIDTH = 96
HEIGHT = 64


class BrainDQN:
    def __init__(
            self,
            n_actions,
            learning_rate=0.001,
            reward_decay=0.99,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=50000,
            batch_size=32,
            e_greedy_increment=0.0001,
            output_graph=True,
            dueling=False,
            sess=None,
    ):
        self.frame_per_action = FRAME_PER_ACTION
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.width = WIDTH
        self.height = HEIGHT
        self.state = None

        self.dueling = dueling  # decide to use dueling DQN or not

        self.learn_step_counter = 0
        self.n_features = self.width * self.height * 4
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 3))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            self.summary_writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

        self.saver = tf.train.Saver(max_to_keep=1)
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('conv1'):
                w_conv1 = tf.get_variable('w_conv1', [8, 8, 4, 6], initializer=w_initializer, collections=c_names)
                b_conv1 = tf.get_variable('b_conv1', [1, 6], initializer=w_initializer, collections=c_names)
                conv1 = tf.nn.relu(tf.nn.conv2d(s, w_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1)
                pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            with tf.variable_scope('conv2'):
                w_conv2 = tf.get_variable('w_conv2', [8, 8, 6, 3], initializer=w_initializer, collections=c_names)
                b_conv2 = tf.get_variable('b_conv2', [1, 3], initializer=w_initializer, collections=c_names)
                conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
                pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
                conv2 = tf.reshape(pool2, (-1, 288))
            # with tf.variable_scope('conv3'):
            #     w_conv3 = tf.get_variable('w_conv3', [3, 3, 32, 64], initializer=w_initializer, collections=c_names)
            #     b_conv3 = tf.get_variable('b_conv3', [1, 64], initializer=w_initializer, collections=c_names)
            #     conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w_conv3, strides=[1, 2, 2, 1], padding='SAME') + b_conv3)
            #     conv3 = tf.reshape(conv3, [-1, 1536])
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [288, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(conv2, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
            else:
                w2_a = tf.get_variable('w2_a', [n_l1, self.n_actions], initializer=w_initializer,
                                       collections=c_names)
                b2_a = tf.get_variable('b2_a', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out_a = tf.matmul(l1, w2_a) + b2_a

                w2_d = tf.get_variable('w2_d', [n_l1, self.n_actions], initializer=w_initializer,
                                       collections=c_names)
                b2_d = tf.get_variable('b2_d', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out_d = tf.matmul(l1, w2_d) + b2_d

                out = out_a + out_d
            return out, out_a, out_d

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.width, self.height, 4], name='s')  # input
        self.q_input_a = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_input_a')  # for calculating loss
        self.q_input_d = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_input_d')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 80, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval, self.q_eval_a, self.q_eval_d = build_layers(self.s, c_names, n_l1, w_initializer,
                                                                     b_initializer)

        with tf.variable_scope('loss'):
            self.loss_a = tf.squared_difference(self.q_input_a, self.q_eval_a)
            self.loss_d = tf.squared_difference(self.q_input_d, self.q_eval_d)
            self.loss = tf.reduce_mean(self.loss_a + self.loss_d)
            tf.summary.scalar("cost", self.loss)
            tf.summary.scalar("cost_attack", tf.reduce_mean(self.loss_a))
            tf.summary.scalar("cost_defence", tf.reduce_mean(self.loss_d))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.width, self.height, 4], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_evalT, self.q_evalT_a, self.q_evalT_d = build_layers(self.s_, c_names, n_l1, w_initializer,
                                                                        b_initializer)
        self.merged = tf.summary.merge_all()

    def first_store(self, s):
        s = s.reshape((self.width, self.height))
        self.state = np.stack((s, s, s, s), axis=2)

    def store_transition(self, a, r, s_):
        r_a = r[0] / REWARD_MAX
        r_d = r[1] / REWARD_MAX
        current_state = self.state
        self.state = np.append(self.state[:, :, 1:], s_, axis=2)

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack(
            (current_state.reshape(self.n_features), [a, r_a, r_d], self.state.reshape(self.n_features)))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def get_action(self, observation):
        # observation = observation[np.newaxis, :]
        state = self.state
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            action = self.sess.run(self.q_eval, feed_dict={self.s: state})
            # action = np.argmax(actions_value)
        else:
            action = np.zeros(self.n_actions)
            action_indx = random.randrange(self.n_actions)
            action[action_indx] = 1
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        memory_state = batch_memory[:, :self.n_features]
        memory_state_ = batch_memory[:, -self.n_features:]
        memory_state = memory_state.reshape((self.batch_size, self.width, self.height, 4))
        memory_state_ = memory_state_.reshape((self.batch_size, self.width, self.height, 4))

        q_evalT_a, q_evalT_d, q_eval_a, q_eval_d = self.sess.run(
            [self.q_evalT_a, self.q_evalT_d, self.q_eval_a, self.q_eval_d],
            feed_dict={self.s_: memory_state_,
                       self.s: memory_state})

        q_input_a = q_eval_a.copy()
        q_input_d = q_eval_d.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward_a = batch_memory[:, self.n_features + 1]
        reward_d = batch_memory[:, self.n_features + 2]

        q_input_a[batch_index, eval_act_index] = reward_a + self.gamma * np.max(q_evalT_a, axis=1)
        q_input_d[batch_index, eval_act_index] = reward_d + self.gamma * np.max(q_evalT_d, axis=1)

        _, self.cost, merged = self.sess.run([self._train_op, self.loss, self.merged],
                                             feed_dict={self.s: memory_state,
                                                        self.q_input_a: q_input_a,
                                                        self.q_input_d: q_input_d})
        self.summary_writer.add_summary(merged, self.learn_step_counter)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        if self.learn_step_counter % SAVE_AFTER_STEP == 0:
            self.saver.save(self.sess, 'saved_networks/' + 'network' + '-ddqn-' + str(self.learn_step_counter))
        print("Step: ", self.learn_step_counter)
        print("Epsilon: ", round(self.epsilon, 6))
        print("Cost: ", self.cost)
