from env import KnapsackEnvironment
from benchmark_policies import greedy
import numpy as np

DECISION_POINTS = 10
KNAPSACK_CAPACITY = 0.5

# initialize knapsack environment
knapsack_problem = KnapsackEnvironment(n_decision_points=DECISION_POINTS,
                                       knapsack_capacity=KNAPSACK_CAPACITY)

scores = []  # track total profit for each instance
for instance_id in range(0, 100):  # test a policy on 100 knapsack instances
    observation = knapsack_problem.reset()  # initialize a new instance and return first observation (state 0)
    while True:  # as long as the instance has not terminated
        action = greedy(observation)  # choose an action according to the policy
        observation, reward, done = knapsack_problem.step(action)  # transition to the next state
        if done:  # if instance is over
            scores.append(knapsack_problem.total_profit)  # track achieved total profit
            break

print("Mean total profit over all instances {:.3f}.".format(np.mean(scores)))
print("Maximum total profit over all instances {:.3f}.".format(np.max(scores)))