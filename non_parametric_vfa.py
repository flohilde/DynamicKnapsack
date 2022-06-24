import numpy as np


class QTable:
    """
    Q-table policy that fits a Q-table based on offline simulations.
    """

    def __init__(self, k_decision_points: int, aggregation_steps: int, knapsack_capacity: float):
        """

        :param k_decision_points: Number of decision points (total number of items).
        :param aggregation_steps: In how many steps do we approximate the capacity of our knapsack.
        :param knapsack_capacity: The absolute capacity of the knapsack.
        """
        super(QTable, self).__init__()

        self.qtable = np.zeros(shape=(k_decision_points, aggregation_steps))  # initialize Q-table
        self.update_counter = np.zeros(shape=(k_decision_points, aggregation_steps))  # track updates in Q-table

        # auxiliary information
        self.knapsack_capacity = knapsack_capacity  # starting capacity of our knapsack
        self.qtable_shape = (k_decision_points, aggregation_steps)  # shape of our knapsack

    def act(self, observation):
        """
        Derive a decision based on the current state and our current Q-table.

        :param observation: Observation as returned by the KnapsackEnvironment
        :return: Returns a decision (True: We take the given item, False: We do not take the given item).
        """
        # unpack observation
        b = observation["b"]  # current knapsack capacity
        item = observation["item"]  # weight of the presented item
        p = observation["p"]  # value of the presented item
        ct = observation["t"]  # current decision point

        if b < item:  # if the item does not fit, we return that we do not take it
            return False

        # Else we calulate both possible post-decision state values:

        # case 1: Taking the item
        # calculate the remaining knapsack capacity (in percent) if we take the item
        remaining_b_taking = (b - item) / self.knapsack_capacity
        # look up the value of the PDS (given by decision point and remaining capacity) in the Q-table
        column_index = min(int(self.qtable_shape[1] * remaining_b_taking), self.qtable_shape[1] - 1)
        row_index = ct
        pds_value_taking = self.qtable[row_index, column_index]

        # case 2: Not taking the item
        # calculate the remaining knapsack capacity (in percent) if we don't take the item
        remaining_b_not_taking = b / self.knapsack_capacity
        # look up the value of the PDS in the Q-table
        column_index = min(int(self.qtable_shape[1] * remaining_b_not_taking), self.qtable_shape[1] - 1)
        row_index = ct
        pds_value_not_taking = self.qtable[row_index, column_index]

        # solve bellman equation:
        if p + pds_value_taking > pds_value_not_taking:
            return True
        return False

    def train(self, env, n_iterations, evaluate_every_n):
        """

        :param env: KnapsackEnvironment to train on.
        :param n_iterations: Number of training iterations to perform.
        :param evaluate_every_n: Test the policy's performance every n iterations for 100 iterations (no exploration)
        :return:
        """
        # loop over the training iterations
        for training_episode in range(n_iterations):
            # calculate the current exploration rate
            current_exploration_rate = 1.0 * np.exp(-1 * training_episode / (n_iterations / 4))

            observed_pds = []  # track the observed post-decision states
            collected_rewards = []  # track the collected rewards

            observation = env.reset()  # start a new instance and get the first state observation
            while True:  # while the instance is not over
                # either get an action via exploration or exploitation:
                if np.random.random() < current_exploration_rate:  # exploration
                    if observation["b"] >= observation["item"]:  # only if we can take the item
                        action = np.random.random() > 0.5  # flip a coin whether to take or not
                    else:
                        action = False
                else:  # exploitation
                    action = self.act(observation)  # choose an action according to the current Q-table

                observation, reward, done = env.step(action)  # transition to the next state
                observed_pds.append((observation["t"] - 1, observation["b"]))  # collect the observed PDS
                collected_rewards.append(reward)  # collect the obtained reward

                if done:  # if instance is over
                    self._update(observed_pds, collected_rewards)  # update the Q-table

                    if training_episode % evaluate_every_n == 0:  # evaluate the policy on 100 instances
                        current_performance = self._evaluate(env, 100)
                        perc_explored = np.sum(self.update_counter != 0) / (self.qtable_shape[0] * self.qtable_shape[1])
                        print("Episode: {}\t Explored: {:.2f}\t "
                              "Mean Profit {:.2f}\t Exploration Rate: {:.2f}.".format(training_episode,
                                                                                      perc_explored,
                                                                                      current_performance,
                                                                                      current_exploration_rate))
                    break

    def _update(self, observed_pds, collected_rewards):
        """
        Internal method to update the Q-table according to Q = (1-\alpha)Q_old + \alpha Q_new.
        We choose \alpha=1 / sqrt(n), where n is the number of times we have seen the PDS.

        :param observed_pds: List of observed post-decision states.
        :param collected_rewards: List of collected rewards
        :return: Returns nothing but updates the Q-table in place.
        """
        collected_rewards.pop(0)  # delete the first reward as it does not belong to a pds
        collected_rewards.append(0.0)  # append a reward of 0.0 for the last pds
        # calculate the value of each pds according to the sum of rewards obtained after the PDS.
        pds_values = np.cumsum(np.array(collected_rewards)[::-1])[::-1]
        # Loop over PDS and PDS-values and update Q-table according to update formula.
        for pds, value in zip(observed_pds, pds_values):
            # compute Q-table cell of the given PDS
            row_index = pds[0]
            col_index = min(int(self.qtable_shape[1] * (pds[1]/self.knapsack_capacity)), self.qtable_shape[1] - 1)
            # update n in our counter-table and use it to compute \alpha
            self.update_counter[row_index, col_index] += 1
            alpha = 1 / np.sqrt(self.update_counter[row_index, col_index])
            # update Q-table
            q_old = self.qtable[row_index, col_index]
            q_new = value
            self.qtable[row_index, col_index] = (1 - alpha) * q_old + alpha * q_new

    def _evaluate(self, env, n_iterations):
        """
        Internal method to evaluate the current Q-table policy on n_iterations.

        :param env: Environment to evaluate on.
        :param n_iterations: Number of iterations used for evaluation.
        :return: List of total rewards for each instance.
        """
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









