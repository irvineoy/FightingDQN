"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)
FRAME_PER_ACTION = 5
REWARD_MAX = 40.
SAVE_AFTER_STEP = 10000
NEURALSIZE = 80


class BrainDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.1,
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
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling  # decide to use dueling DQN or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
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
            tf.summary.FileWriter("logs/", self.sess.graph)
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
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

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
                with tf.variable_scope('Q'):
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
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target_a = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.q_target_d = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], NEURALSIZE, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval, self.q_eval_a, self.q_eval_d = build_layers(self.s, c_names, n_l1, w_initializer,
                                                                     b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target_a, self.q_eval_a) +
                                       tf.squared_difference(self.q_target_d, self.q_eval_d))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next, self.q_next_a, self.q_next_d = build_layers(self.s_, c_names, n_l1, w_initializer,
                                                                     b_initializer)

    def store_transition(self, s, a, r, s_):
        r_a = r[0] / REWARD_MAX
        r_d = r[1] / REWARD_MAX
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r_a, r_d], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def get_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # next observation
        q_next_a, q_next_d, q_eval_a, q_eval_d = self.sess.run(
            [self.q_next_a, self.q_next_d, self.q_eval_a, self.q_eval_d],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],
                       self.s: batch_memory[:, :self.n_features]})
        # q_next_d = self.sess.run(self.q_next_d, feed_dict={self.s_: batch_memory[:, -self.n_features:]})
        # q_eval_a = self.sess.run(self.q_eval_a, {self.s: batch_memory[:, :self.n_features]})
        # q_eval_d = self.sess.run(self.q_eval_d, {self.s: batch_memory[:, :self.n_features]})

        q_target_a = q_eval_a.copy()
        q_target_d = q_eval_d.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward_a = batch_memory[:, self.n_features + 1]
        reward_d = batch_memory[:, self.n_features + 2]

        q_target_a[batch_index, eval_act_index] = reward_a + self.gamma * np.max(q_next_a, axis=1)
        q_target_d[batch_index, eval_act_index] = reward_d + self.gamma * np.max(q_next_d, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target_a: q_target_a,
                                                self.q_target_d: q_target_d})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        if self.learn_step_counter % SAVE_AFTER_STEP == 0:
            self.saver.save(self.sess, 'saved_networks/' + 'network' + '-ddqn-' + str(self.learn_step_counter))
        print("Step: ", self.learn_step_counter)
        print("Epsilon: ", round(self.epsilon, 6))
        print("Cost: ", self.cost)
