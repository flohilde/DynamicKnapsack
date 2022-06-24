class Greedy:
    """
    Greedy policy that takes an item if it fits into the knapsack.
    """

    def __init__(self):
        pass

    @staticmethod
    def act(observation):
        """
        Returns True if we can pack the item and False if not.
        :param observation: Current Observation
        :return: Decision as given by True (we pack the item) and False (we do not pack the item).
        """
        if observation["b"] > observation["item"]:  # test if item fits in the knapsack
            return True
        return False
