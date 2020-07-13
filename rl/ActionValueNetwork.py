import os
from datetime import datetime

import tensorflow as tf


class ActionValueNetwork(object):
    def __init__(self, num_inputs, num_actions, learning_rate, name, log=False, load=True):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.loss = tf.keras.metrics.Mean("loss")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.name = name
        self.checkpoint_path = "model_checkpoint"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        self.model = self.load_model() if load else self.create(name)

        if log:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = 'logs/gradient_tape/' + current_time + '/'
            self.summary_writer = tf.summary.create_file_writer(log_dir)

    def create(self, name):
        model = tf.keras.Sequential(name=name)
        model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(self.num_inputs,)))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(self.num_actions))
        model.compile(self.optimizer, loss="mse")
        return model

    def update_weights(self, target_weights):
        self.model.set_weights(target_weights)

    def predict(self, state):
        if self.model is None:
            raise Exception("Null Value, Model not created")
        return self.model(state)

    def checkpoint(self, episode):
        self.model.save(f'{self.checkpoint_path}/{self.name.lower()}/full_model')
        if episode % 20 == 0:
            self.model.save_weights(f'{self.checkpoint_path}/{self.name.lower()}/weights/weights_episode_{episode}')

    def load_model(self, episode=None):
        model = tf.keras.models.load_model(f'{self.checkpoint_path}/{self.name.lower()}/full_model')
        if episode is not None:
            model.load_weights(f'{self.checkpoint_path}/{self.name.lower()}/weights/weights_episode_{episode}')
        return model

