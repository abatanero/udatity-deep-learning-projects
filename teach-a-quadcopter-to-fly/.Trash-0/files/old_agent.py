import numpy as np
from task import Task
from model import QNetwork

class DQN_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1
        
        # Exploration parameters
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability 
        self.decay_rate = 0.0001            # exponential decay rate for exploration prob

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state, step, mainQN, sess):
        # Explore or Exploit
        explore_p = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*step) 
        if explore_p > np.random.rand():
            # Make a random action
            random_w = np.random.normal(
                size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
                scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range
            action = np.dot(state, random_w)
        else:
            # Get action from Q-network
            feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
            Qs = sess.run(mainQN.output, feed_dict=feed)
            action = Qs[0]
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        