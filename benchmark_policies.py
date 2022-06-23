class Greedy:
    """
    Greedy policy that takes an item if it fits into the knapsack.
    """

    def __init__(self):
        pass

    @staticmethod
    def act(observation):
        if observation["b"] > observation["item"]:
            return True
        return False
