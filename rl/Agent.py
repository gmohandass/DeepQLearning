import tensorflow as tf
import numpy as np
import gym
import random

from rl.ActionValueNetwork import ActionValueNetwork
from rl.ExperiencedReplay import ExperiencedReplay
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, env,
                 total_episodes=200,
                 total_steps=200,
                 training_batch_size=128,
                 max_buffer_size=20000,
                 learning_rate=0.001,
                 target_update_freq=10,
                 gamma=0.99):

        self.env = env
        self.total_episodes: int = total_episodes
        self.total_steps: int = total_steps
        self.target_update_freq: int = target_update_freq
        self.gamma: tf.Tensor = tf.constant(gamma, dtype=tf.float32)
        self.num_inputs, self.num_actions = self.env.observation_space.shape[0] + 1, self.env.action_space.n
        self.q_network = ActionValueNetwork(self.num_inputs, self.num_actions, learning_rate, "Q-Network", log=True)
        self.target_q_network = ActionValueNetwork(self.num_inputs, self.num_actions, learning_rate, "Target-Q-Network")
        self.replay_buffer = ExperiencedReplay(max_buffer_size, training_batch_size)
        self.avg_rewards = []

    def get_epsilon(self, episode):
        episode_percent = episode / self.total_episodes
        if episode_percent > 0.5:
            return 0.1
        # elif episode_percent > 0.7:
        #     return 0.2
        # elif episode_percent > 0.5:
        #     return 0.4
        # elif episode_percent > 0.3:
        #     return 0.6
        # elif episode_percent > 0.1:
        #     return 0.8
        else:
            return 0.7

    @staticmethod
    def normalize(obs, time):
        position, velocity = obs
        min_position = -1.2
        max_position = 0.6
        min_velocity = -0.07
        max_velocity = 0.07
        min_time = 0
        max_time = 200
        position_norm = (position - min_position) / (max_position - min_position)
        velocity_norm = (velocity - min_velocity) / (max_velocity - min_velocity)
        time_norm = (time - min_time) / (max_time - min_time)
        observations = [position_norm, velocity_norm, time_norm]
        return np.array([observations])

    @staticmethod
    def _to_tensor(observations):
        return tf.reshape(tf.convert_to_tensor(observations[0], dtype=tf.float32), (1, observations[0].shape[0]))

    def _select_action(self, episode, current_state):
        q_pred = self.q_network.predict(self._to_tensor(current_state))
        epsilon = self.get_epsilon(episode)
        action_type = "Exploit"
        action = tf.argmax(q_pred, axis=1).numpy()[0]

        if random.random() < epsilon:
            action_type = "Explore"
            action = env.action_space.sample()

        return action, action_type

    def show_state(self, env, action, action_type, epsilon, rewards, episode=0, step=0, info=""):
        sum_reward = sum(rewards)
        avg_reward = sum_reward / len(rewards)
        if step == 199:
            self.avg_rewards.append(avg_reward)

        plt.figure(3)
        plt.clf()
        plt.imshow(env.render(mode='rgb_array'))
        plt.title(f"Mountain Car | Episode: {episode}/{self.total_episodes} | "
                  f"Step: {step} | Action: {action} | ActionType: {action_type} | "
                  f"Epsilon: {epsilon} | Total Reward: {sum_reward} | Avg Reward: {avg_reward}")
        plt.axis('off')

    def learn(self):
        sample_state, sample_action, sample_reward, sample_new_state, sample_done = self.replay_buffer.sample()
        sample_actions_encoded = tf.squeeze(tf.one_hot(sample_action, self.num_actions, on_value=1.0, off_value=0.0))

        future_q_val = tf.expand_dims(tf.reduce_max(self.target_q_network.predict(sample_new_state), axis=1), axis=1)
        target_q = sample_reward + (self.gamma * future_q_val * (1 - sample_done))

        with tf.GradientTape() as tape:
            q_pred = tf.reduce_sum(self.q_network.predict(sample_state) * sample_actions_encoded, axis=-1)
            loss = tf.keras.losses.MSE(target_q, q_pred)

        grads = tape.gradient(loss, self.q_network.model.trainable_weights)
        self.q_network.optimizer.apply_gradients(zip(grads, self.q_network.model.trainable_weights))
        self.q_network.loss(loss)

    def act(self):

        for episode in range(self.total_episodes):
            observation = self.env.reset()
            current_state = self.normalize(observation, 0)
            rewards = []
            for step in range(1, self.total_steps + 1):
                action, action_type = self._select_action(episode, current_state)
                observation, reward, done, info = self.env.step(action)
                reward = reward * step + observation[1] + 10 * abs(observation[0])
                if observation[0] >= 0.45:
                    print(f"Success {episode}")
                    reward += 100000000
                new_state = self.normalize(observation, step)
                self.replay_buffer.add_exp(current_state,
                                           action,
                                           reward,
                                           new_state,
                                           done)

                current_state = new_state
                rewards.append(reward)
                self.learn()
                if episode % self.target_update_freq == 0 & step == 1:
                    print("Updated target Q network")
                    self.target_q_network.update_weights(self.q_network.model.get_weights())

                self.show_state(env, action, action_type, self.get_epsilon(episode), rewards, episode, step)

                with self.q_network.summary_writer.as_default():
                    tf.summary.scalar("loss", self.q_network.loss.result(), step=step)

                if done:
                    print(f"Mountain Car | Episode: {episode}/{self.total_episodes} | "
                          f"Epsilon: {self.get_epsilon(episode)} | Total Reward: {sum(rewards)} "
                          f"| Avg Reward: {sum(rewards)/self.total_steps}")
                    break

            self.q_network.checkpoint(episode)
            self.target_q_network.checkpoint(episode)
            self.q_network.loss.reset_states()

        self.env.close()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = Agent(env,
                  total_episodes=200,
                  total_steps=201,
                  training_batch_size=128,
                  max_buffer_size=20000,
                  learning_rate=0.001,
                  target_update_freq=10,
                  gamma=0.99)

    agent.act()
