def greedy(observation):
    """
    Greedy policy that takes an item if it fits into the knapsack.
    """
    if observation["b"] > observation["item"]:
        return True
    return False
