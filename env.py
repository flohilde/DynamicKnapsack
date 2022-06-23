import numpy as np


class KnapsackEnvironment:

    def __init__(self, n_decision_points, knapsack_capacity):
        super(KnapsackEnvironment, self).__init__()

        # general parameters
        self.t = n_decision_points  # number of decision points (items) in an instance
        self.alpha = knapsack_capacity  # knapsack capacity relative to the total capacity of all items

        # state
        self.total_profit = None            # total collected profit over the instance
        self.ct = None                      # current time step in (0,...,n_decision_points)
        self.items = None                   # all sampled items for the instance
        self.p = None                       # all sampled profits for the items of the instance
        self.b = None                       # initial capacity of our knapsack
        self.current_b = None               # current capacity of our knapsack
        self.current_observation = None     # current state given by revealed item, its profit,
                                            # the remainining knapsack capacity and the current time point

    @property
    def observation(self):
        """
        Returns the state information in the form of a dictionary that contains the current knapsack capacity,
        the revealed profit of the item, the revealed item, and the current time point (in this order).
        """
        revealed_p = self.p[self.ct]
        revealed_item = self.items[self.ct]
        self.current_observation = {"b": self.current_b, "p": revealed_p, "item": revealed_item, "t": self.ct}
        return self.current_observation

    @property
    def done(self):
        """
        Returns True if the instance is over, else False.
        """
        if self.ct == self.t:  # termination criterium (time-over )
            return True
        if (self.b < self.items).all():  # stronger termination criterium (no remaining item fits into the knapsack)
            return True
        return False

    def step(self, action):
        """
        Transitions from one state to the next depending on the action taken and the stochastic information.
        """

        assert isinstance(action, bool)  # assert that the action is a boolean
        b = self.current_observation["b"]  # current knapsack capacity
        p = self.current_observation["p"]  # profit of current item
        item = self.current_observation["item"]  # current item

        reward = 0.0  # initialize reward for this state
        if action:  # if we take the item
            assert b > item  # make sure the item fits into the knapsack
            reward = p  # update reward according to the items profit
            self.current_b -= item  # update the capacity of our knapsack by deducting the current item
            self.total_profit += reward  # update the total reward collected
        self.ct += 1  # move to the next decision point

        # we return the next state observation, the reward, and whether the instance is over
        if self.done:  # if decision point was last decision point return only the profit and no next state
            return None, reward, True
        return self.observation, reward, False  # else we return the next state observation and the profit

    def reset(self):
        """
        Samples a new instance by sampling items and profits and constructing the knapsack accordingly.
        """
        self.ct = 0  # ct now stars at 0 and goes to n_decision_points
        self.total_profit = 0  # initialize total profit to 0

        self.items = np.random.uniform(size=self.t)  # sample items
        self.items = self.items / np.sum(self.items)  # normalize items

        self.b = self.alpha  # initialize knapsack to a fraction of total item capacity given by alpha
        self.current_b = self.b  # initialize current knapsack

        # Calculate the value pj of the items according to Chu & Beasley
        self.p = self.items + 0.5 * np.random.uniform(size=self.t)
        self.p = self.p / np.sum(self.p)  # normalize profits

        return self.observation  # return the observation of the first state
