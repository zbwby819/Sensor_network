# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:46:21 2020

@author: azrael
"""
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.tools import freeze_graph


def create_agent_model(env, lr=1e-4, h_size=128, epsilon=0.2, beta=1e-3, max_step=5e6, normalize=0, num_layers=2):
    """
    Takes a Unity environment and model-specific hyper-parameters and returns the
    appropriate PPO agent model for the environment.
    :param env: a Unity environment.
    :param lr: Learning rate.
    :param h_size: Size of hidden layers/
    :param epsilon: Value for policy-divergence threshold.
    :param beta: Strength of entropy regularization.
    :return: a sub-class of PPOAgent tailored to the environment.
    :param max_step: Total number of training steps.
    """
    if num_layers < 1:
        num_layers = 1

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    if brain.action_space_type == "continuous":
        return ContinuousControlModel(lr, brain, h_size, epsilon, max_step, normalize, num_layers)
    if brain.action_space_type == "discrete":
        return DiscreteControlModel(lr, brain, h_size, epsilon, beta, max_step, normalize, num_layers)


def save_model(sess, saver, model_path="./", steps=0):
    """
    Saves current model to checkpoint folder.
    :param sess: Current Tensorflow session.
    :param model_path: Designated model path.
    :param steps: Current number of steps in training process.
    :param saver: Tensorflow saver for session.
    """
    last_checkpoint = model_path + '/model-' + str(steps) + '.cptk'
    saver.save(sess, last_checkpoint)
    tf.train.write_graph(sess.graph_def, model_path, 'raw_graph_def.pb', as_text=False)
    print("Saved Model")


def export_graph(model_path, env_name="env", target_nodes="action,value_estimate,action_probs"):
    """
    Exports latest saved model to .bytes format for Unity embedding.
    :param model_path: path of model checkpoints.
    :param env_name: Name of associated Learning Environment.
    :param target_nodes: Comma separated string of needed output nodes for embedded graph.
    """
    ckpt = tf.train.get_checkpoint_state(model_path)
    freeze_graph.freeze_graph(input_graph=model_path + '/raw_graph_def.pb',
                              input_binary=True,
                              input_checkpoint=ckpt.model_checkpoint_path,
                              output_node_names=target_nodes,
                              output_graph=model_path + '/' + env_name + '.bytes',
                              clear_devices=True, initializer_nodes="", input_saver="",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")


class PPOModel(object):
    def __init__(self):
        self.normalize = 0

    def _create_global_steps(self):
        """Creates TF ops to track and increment global training step."""
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
        self.increment_step = tf.assign(self.global_step, tf.cast(self.global_step, tf.int32) + 1)

    def _create_reward_encoder(self):
        """Creates TF ops to track and increment recent average cumulative reward."""
        self.last_reward = tf.Variable(0, name="last_reward", trainable=False, dtype=tf.float32)
        self.new_reward = tf.placeholder(shape=[], dtype=tf.float32, name='new_reward')
        self.update_reward = tf.assign(self.last_reward, self.new_reward)

    def _create_visual_encoder(self, o_size_h, o_size_w, bw, h_size, num_streams, activation, num_layers):
        """
        Builds a set of visual (CNN) encoders.
        :param o_size_h: Height observation size.
        :param o_size_w: Width observation size.
        :param bw: Whether image is greyscale {True} or color {False}.
        :param h_size: Hidden layer size.
        :param num_streams: Number of visual streams to construct.
        :param activation: What type of activation function to use for layers.
        :return: List of hidden layer tensors.
        """
        if bw:
            c_channels = 1
        else:
            c_channels = 3

        self.observation_in = tf.placeholder(shape=[None, 8, o_size_h, o_size_w, c_channels], dtype=tf.float32,
                                             name='observation_0')
        streams = []
        for i in range(num_streams):
            self.conv1 = tf.layers.conv3d(self.observation_in, 16, kernel_size=[4, 8, 8], strides=[2, 4, 4],
                                          use_bias=False, activation=activation)
            self.conv2 = tf.layers.conv3d(self.conv1, 32, kernel_size=[3, 4, 4], strides=[1, 2, 2],
                                          use_bias=False, activation=activation)
            # self.conv3 = tf.layers.conv2d(self.conv2, 32, kernel_size=[4, 4], strides=[2, 2],
            #                               use_bias=False, activation=activation)
            hidden = tf.layers.flatten(self.conv2)
            for j in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams

    def _create_resnet_visual_encoder(self, o_size_h, o_size_w, bw, h_size, num_streams, activation, num_layers):
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
        :return: List of hidden layer tensors.
        """
        n_channels = [16, 32, 32]  # channel for each stack
        n_blocks = 2  # number of residual blocks

        if bw:
            c_channels = 1
        else:
            c_channels = 3

        self.observation_in = tf.placeholder(shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32,
                                             name='observation_0')

        streams = []
        for t in range(num_streams):
            for i, ch in enumerate(n_channels):
                hidden = tf.layers.conv2d(
                    self.observation_in,
                    ch,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    reuse=False,
                    name="layer%dconv_1" % i,
                )
                hidden = tf.layers.max_pooling2d(
                    hidden, pool_size=[3, 3], strides=[2, 2], padding="same"
                )
                # create residual blocks
                for j in range(n_blocks):
                    block_input = hidden
                    hidden = tf.nn.relu(hidden)
                    hidden = tf.layers.conv2d(
                        hidden,
                        ch,
                        kernel_size=[3, 3],
                        strides=[1, 1],
                        padding="same",
                        reuse=False,
                        name="layer%d_%d_conv1" % (i, j),
                    )
                    hidden = tf.nn.relu(hidden)
                    hidden = tf.layers.conv2d(
                        hidden,
                        ch,
                        kernel_size=[3, 3],
                        strides=[1, 1],
                        padding="same",
                        reuse=False,
                        name="layer%d_%d_conv2" % (i, j),
                    )
                    hidden = tf.add(block_input, hidden)
            hidden = tf.nn.relu(hidden)
            hidden = tf.layers.flatten(hidden)

            for i in range(num_layers):
                hidden = tf.layers.dense(
                    hidden,
                    h_size,
                    activation=activation,
                    reuse=False,
                    name="hidden_{}".format(i),
                    kernel_initializer=tf.initializers.variance_scaling(1.0),
                )

            streams.append(hidden)

        return streams

    def _create_continuous_state_encoder(self, s_size, h_size, num_streams, activation, num_layers):
        """
        Builds a set of hidden state encoders.
        :param s_size: state input size.
        :param h_size: Hidden layer size.
        :param num_streams: Number of state streams to construct.
        :param activation: What type of activation function to use for layers.
        :return: List of hidden layer tensors.
        """
        self.state_in = tf.placeholder(shape=[None, s_size[0], s_size[1], s_size[2]],
                                       dtype=tf.float32, name='state')

        if self.normalize > 0:
            self.running_mean = tf.get_variable("running_mean", [s_size], trainable=False, dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
            self.running_variance = tf.get_variable("running_variance", [s_size], trainable=False, dtype=tf.float32,
                                                    initializer=tf.ones_initializer())
            self.norm_running_variance = tf.get_variable("norm_running_variance", [s_size], trainable=False,
                                                         dtype=tf.float32,
                                                         initializer=tf.ones_initializer())

            self.normalized_state = tf.clip_by_value(
                (self.state_in - self.running_mean) / tf.sqrt(self.norm_running_variance), -5, 5, name="normalized_state")

            self.new_mean = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_mean')
            self.new_variance = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_variance')
            self.update_mean = tf.assign(self.running_mean, self.new_mean)
            self.update_variance = tf.assign(self.running_variance, self.new_variance)
            self.update_norm_variance = tf.assign(self.norm_running_variance,
                                                  self.running_variance / (tf.cast(self.global_step, tf.float32) + 1))
        else:
            self.normalized_state = self.state_in
        streams = []
        for i in range(num_streams):
            hidden = self.normalized_state
            for j in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams

    def _create_discrete_state_encoder(self, s_size, h_size, num_streams, activation, num_layers):
        """
        Builds a set of hidden state encoders from discrete state input.
        :param s_size: state input size (discrete).
        :param h_size: Hidden layer size.
        :param num_streams: Number of state streams to construct.
        :param activation: What type of activation function to use for layers.
        :return: List of hidden layer tensors.
        """
        self.state_in = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='state')
        state_in = tf.reshape(self.state_in, [-1])
        state_onehot = tf.one_hot(state_in, s_size)
        streams = []
        hidden = state_onehot
        for i in range(num_streams):
            for j in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams

    def _create_ppo_optimizer(self, probs, old_probs, value, entropy, beta, epsilon, lr, max_step):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value: Current value estimate
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        """

        self.returns_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='discounted_rewards')
        self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantages')

        decay_epsilon = tf.train.polynomial_decay(epsilon, self.global_step,
                                                  max_step, 1e-2,
                                                  power=1.0)

        r_theta = probs / (old_probs + 1e-10)
        p_opt_a = r_theta * self.advantage
        p_opt_b = tf.clip_by_value(r_theta, 1 - decay_epsilon, 1 + decay_epsilon) * self.advantage
        self.policy_loss = -tf.reduce_mean(tf.minimum(p_opt_a, p_opt_b))

        self.value_loss = tf.reduce_mean(tf.squared_difference(self.returns_holder,
                                                               tf.reduce_sum(value, axis=1)))

        decay_beta = tf.train.polynomial_decay(beta, self.global_step,
                                               max_step, 1e-5,
                                               power=1.0)
        self.loss = self.policy_loss + self.value_loss - decay_beta * tf.reduce_mean(entropy)

        self.learning_rate = tf.train.polynomial_decay(lr, self.global_step,
                                                       max_step, 1e-10,
                                                       power=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update_batch = optimizer.minimize(self.loss)


class ContinuousControlModel(PPOModel):
    def __init__(self, lr, brain, h_size, epsilon, max_step, normalize, num_layers):
        """
        Creates Continuous Control Actor-Critic model.
        :param brain: State-space size
        :param h_size: Hidden layer size
        """
        super(ContinuousControlModel, self).__init__()
        s_size = brain.state_space_size
        a_size = brain.action_space_size

        self.normalize = normalize
        self._create_global_steps()
        self._create_reward_encoder()

        hidden_state, hidden_visual, hidden_policy, hidden_value = None, None, None, None
        if brain.number_observations > 0:
            height_size, width_size = brain.camera_resolutions[0]['height'], brain.camera_resolutions[0]['width']
            bw = brain.camera_resolutions[0]['blackAndWhite']
            hidden_visual = self._create_visual_encoder(height_size, width_size, bw, h_size, 2, tf.nn.tanh, num_layers)
        if brain.state_space_size is not None:
            s_size = brain.state_space_size
            if brain.state_space_type == "continuous":
                hidden_state = self._create_continuous_state_encoder(s_size, h_size, 2, tf.nn.tanh, num_layers)
            else:
                hidden_state = self._create_discrete_state_encoder(s_size, h_size, 2, tf.nn.tanh, num_layers)

        if hidden_visual is None and hidden_state is None:
            raise Exception("No valid network configuration possible. "
                            "There are no states or observations in this brain")
        elif hidden_visual is not None and hidden_state is None:
            hidden_policy, hidden_value = hidden_visual
        elif hidden_visual is None and hidden_state is not None:
            hidden_policy, hidden_value = hidden_state
        elif hidden_visual is not None and hidden_state is not None:
            hidden_policy = tf.concat([hidden_visual[0], hidden_state[0]], axis=1)
            hidden_value = tf.concat([hidden_visual[1], hidden_state[1]], axis=1)

        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')

        self.mu = tf.layers.dense(hidden_policy, a_size, activation=None, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.01))
        self.log_sigma_sq = tf.get_variable("log_sigma_squared", [a_size], dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
        self.sigma_sq = tf.exp(self.log_sigma_sq)

        self.epsilon = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='epsilon')

        self.output = self.mu + tf.sqrt(self.sigma_sq) * self.epsilon
        self.output_max = tf.identity(self.mu, name='action_max')
        self.output = tf.identity(self.output, name='action')

        a = tf.exp(-1 * tf.pow(tf.stop_gradient(self.output) - self.mu, 2) / (2 * self.sigma_sq))
        b = 1 / tf.sqrt(2 * self.sigma_sq * np.pi)
        self.probs = tf.multiply(a, b, name="action_probs")

        self.entropy = tf.reduce_sum(0.5 * tf.log(2 * np.pi * np.e * self.sigma_sq))

        self.value = tf.layers.dense(hidden_value, 1, activation=None, use_bias=False)
        self.value = tf.identity(self.value, name="value_estimate")

        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')

        self._create_ppo_optimizer(self.probs, self.old_probs, self.value, self.entropy, 0.0, epsilon, lr, max_step)


class DiscreteControlModel(PPOModel):
    def __init__(self, lr, brain, h_size, epsilon, beta, max_step, normalize, num_layers):
        """
        Creates Discrete Control Actor-Critic model.
        :param brain: State-space size
        :param h_size: Hidden layer size
        """
        super(DiscreteControlModel, self).__init__()
        self._create_global_steps()
        self._create_reward_encoder()
        self.normalize = normalize

        hidden_state, hidden_visual, hidden = None, None, None
        if brain.number_observations > 0:
            height_size, width_size = brain.state_space_size[0], brain.state_space_size[1]
            bw = brain.state_space_size[2] == 1
            hidden_visual = self._create_visual_encoder(height_size, width_size, bw, h_size, 1, tf.nn.elu, num_layers)[
                0]

        '''
        if brain.state_space_size is not None:
            s_size = brain.state_space_size
            if brain.state_space_type == "continuous":
                hidden_state = self._create_continuous_state_encoder(s_size, h_size, 1, tf.nn.elu, num_layers)[0]
            else:
                hidden_state = self._create_discrete_state_encoder(s_size, h_size, 1, tf.nn.elu, num_layers)[0]
        '''

        if hidden_visual is None and hidden_state is None:
            raise Exception("No valid network configuration possible. "
                            "There are no states or observations in this brain")
        elif hidden_visual is not None and hidden_state is None:
            hidden = hidden_visual
        elif hidden_visual is None and hidden_state is not None:
            hidden = hidden_state
        elif hidden_visual is not None and hidden_state is not None:
            hidden = tf.concat([hidden_visual, hidden_state], axis=1)

        a_size = brain.action_space_size

        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')
        self.policy = tf.layers.dense(hidden, a_size, activation=None, use_bias=False,
                                      kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.01))
        self.probs = tf.nn.softmax(self.policy, name="action_probs")
        self.output = tf.random.categorical(self.policy, 1)
        self.output_max = tf.argmax(self.probs, name='action_max', axis=1)
        self.output = tf.identity(self.output, name="action")
        self.value = tf.layers.dense(hidden, 1, activation=None, use_bias=False,
                                     kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0))
        self.value = tf.identity(self.value, name="value_estimate")

        self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs + 1e-10), axis=1)

        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = tf.one_hot(self.action_holder, a_size)
        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')
        self.responsible_probs = tf.reduce_sum(self.probs * self.selected_actions, axis=1)
        self.old_responsible_probs = tf.reduce_sum(self.old_probs * self.selected_actions, axis=1)

        self._create_ppo_optimizer(self.responsible_probs, self.old_responsible_probs,
                                   self.value, self.entropy, beta, epsilon, lr, max_step)