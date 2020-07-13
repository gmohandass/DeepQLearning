import numpy as np
import tensorflow as tf


class ExperiencedReplay(object):
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.new_state_buffer = None
        self.done_buffer = None

    @staticmethod
    def _concat(old_array, new_array):
        if not isinstance(new_array, np.ndarray):
            new_array = np.array([[new_array]])

        if old_array is None:
            return new_array
        return np.concatenate([new_array, old_array], axis=0)

    @staticmethod
    def _tf_tensor(np_array, data_type=tf.float32):
        return tf.random.shuffle(tf.convert_to_tensor(np_array, dtype=data_type))

    def add_exp(self, state, action, reward, new_state, done):
        self.state_buffer = self._concat(self.state_buffer, state)
        self.action_buffer = self._concat(self.action_buffer, action)
        self.reward_buffer = self._concat(self.reward_buffer, reward)
        self.done_buffer = self._concat(self.done_buffer, done)
        self.new_state_buffer = self._concat(self.new_state_buffer, new_state)

        if self.state_buffer.shape[0] > self.buffer_size:
            self.state_buffer = self.state_buffer[: self.buffer_size, :]
            self.action_buffer = self.action_buffer[: self.buffer_size, :]
            self.reward_buffer = self.reward_buffer[: self.buffer_size, :]
            self.done_buffer = self.done_buffer[: self.buffer_size, :]
            self.new_state_buffer = self.new_state_buffer[: self.buffer_size, :]

    def sample(self):
        if self.state_buffer.shape[0] > self.batch_size:
            idx = np.random.choice(self.state_buffer.shape[0], self.batch_size, replace=False)
        else:
            idx = list(range(self.state_buffer.shape[0]))

        sample_state_tf = self._tf_tensor(self.state_buffer[idx, :])
        sample_action_tf = self._tf_tensor(self.action_buffer[idx, :], tf.int32)
        sample_reward_tf = self._tf_tensor(self.reward_buffer[idx, :])
        sample_done_tf = self._tf_tensor(self.done_buffer[idx, :])
        sample_new_state_tf = self._tf_tensor(self.new_state_buffer[idx, :])
        return sample_state_tf, sample_action_tf, sample_reward_tf, sample_new_state_tf, sample_done_tf
