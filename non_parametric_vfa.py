import numpy as np


class QTable:

    def __init__(self, k_decision_points: int, aggregation_steps: int, knapsack_capacity: float):
        super(QTable, self).__init__()
        self.knapsack_capacity = knapsack_capacity
        self.qtable_shape = (k_decision_points, aggregation_steps)
        self.qtable = np.zeros(shape=(k_decision_points, aggregation_steps))
        self.update_counter = np.zeros(shape=(k_decision_points, aggregation_steps))  # track updates in Q-table

    def act(self, observation):
        b = observation["b"]
        item = observation["item"]
        p = observation["p"]
        ct = observation["t"]
        if b < item:
            return False
        remaining_b_taking = 1- ((b - item) / self.knapsack_capacity)
        column_index_taking = min(int(self.qtable_shape[1] * remaining_b_taking), self.qtable_shape[1] - 1)
        pds_value_taking = self.qtable[ct, column_index_taking]

        remaining_b_not_taking = 1 - (b / self.knapsack_capacity)
        column_index_not_taking = min(int(self.qtable_shape[1] * remaining_b_not_taking), self.qtable_shape[1] - 1)
        pds_value_not_taking = self.qtable[ct, column_index_not_taking]
        if p + pds_value_taking > pds_value_not_taking:
            return True
        return False

    def fit(self, env, n_iterations, evaluate_every_n):

        for training_episode in range(n_iterations):
            current_exploration_rate = 1.0 * np.exp(-1 * training_episode / (n_iterations / 4))
            observed_pds = []
            collected_rewards = []
            observation = env.reset()
            while True:
                if np.random.random() < current_exploration_rate:  # exploration
                    if observation["b"] >= observation["item"]:  # if we can take the item
                        action = np.random.random() > 0.5  # flip a coin whether to take or not
                    else:
                        action = False
                else:  # exploitation
                    action = self.act(observation)  # choose an action according to the current Q-table
                observation, reward, done = env.step(action)  # transition to the next state
                observed_pds.append((observation["t"] - 1, observation["b"]))
                collected_rewards.append(reward)
                if done:  # if instance is over
                    self._update(observed_pds, collected_rewards)  # update the Q-table

                    if training_episode % evaluate_every_n == 0:
                        current_performance = self._evaluate(env, 100)
                        perc_explored = np.sum(self.update_counter != 0) / (self.qtable_shape[0] * self.qtable_shape[1])
                        print("Episode: {}\t Explored: {:.2f}\t "
                              "Mean Profit {:.2f}\t Exploration Rate: {:.2f}.".format(training_episode,
                                                                                      perc_explored,
                                                                                      current_performance,
                                                                                      current_exploration_rate))
                    break

    def _update(self, observed_pds, collected_rewards):
        collected_rewards.pop(0)  # delete the first reward as it does not belong to a pds
        collected_rewards.append(0.0)  # append a reward of 0.0 for the last pds
        pds_values = np.cumsum(np.array(collected_rewards)[::-1])[::-1]  # calculate the value of each pds
        for pds, value in zip(observed_pds, pds_values):
            row_index = pds[0]
            col_index = min(int(self.qtable_shape[1] * (1 - pds[1]/self.knapsack_capacity)), self.qtable_shape[1] - 1)
            self.update_counter[row_index, col_index] += 1
            alpha = 1 / np.sqrt(self.update_counter[row_index, col_index])
            q_old = self.qtable[row_index, col_index]
            q_new = value
            self.qtable[row_index, col_index] = (1 - alpha) * q_old + alpha * q_new

    def _evaluate(self, env, n_iterations):
        scores = []  # track total profit for each instance
        for instance_id in range(0, n_iterations):  # test a policy on 100 knapsack instances
            observation = env.reset()  # initialize a new instance and return first observation (state 0)
            while True:  # as long as the instance has not terminated
                action = self.act(observation)  # choose an action according to the policy
                observation, reward, done = env.step(action)  # transition to the next state
                if done:  # if instance is over
                    scores.append(env.total_profit)  # track achieved total profit
                    break
        return np.mean(scores)









