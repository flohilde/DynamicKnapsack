from env import KnapsackEnvironment
from benchmark_policies import Greedy
from non_parametric_vfa import QTable
from parametric_vfa import LinearRegression
import numpy as np


def evaluate_policy(env, policy, n_instances):
    """
    Evaluates a given policy on a given Knapsack problem over a given number of instances.
    :param env: KnapsackEnvironment to evaluate on.
    :param policy: Policy to evaluate.
    :param n_instances: Number of instances to simulate.
    :return: List of total rewards for each instance.
    """
    scores = []  # track total profit for each instance
    for instance_id in range(0, n_instances):  # test a policy on 100 knapsack instances
        observation = env.reset()  # initialize a new instance and return first observation (state 0)
        while True:  # as long as the instance has not terminated
            action = policy.act(observation)  # choose an action according to the policy
            observation, reward, done = env.step(action)  # transition to the next state
            if done:  # if instance is over
                scores.append(knapsack_problem.total_profit)  # track achieved total profit
                break
    return scores


if __name__ == "__main__":

    np.random.seed(42)  # set a RNG seed for consisten results

    # set general parameters of our experiment
    DECISION_POINTS = 10
    KNAPSACK_CAPACITY = 0.5
    POLICIES = ["GREEDY", "PARAMETRIC", "NON_PARAMETRIC"]
    POLICY = POLICIES[1]
    N_EVAL_INSTANCES = 1000
    N_TRAIN_INSTANCES = int(1e5)

    # initialize knapsack environment
    knapsack_problem = KnapsackEnvironment(n_decision_points=DECISION_POINTS,
                                           knapsack_capacity=KNAPSACK_CAPACITY)

    if POLICY == "GREEDY":
        greedy_policy = Greedy()
        print("Evaluate Greedy policy on {} instances".format(N_EVAL_INSTANCES))
        scores = evaluate_policy(env=knapsack_problem, policy=greedy_policy, n_instances=100)
        print("Mean total profit over all instances {:.3f}.".format(np.mean(scores)))
        print("Maximum total profit over all instances {:.3f}.".format(np.max(scores)))

    if POLICY == "NON_PARAMETRIC":
        qtable_policy = QTable(k_decision_points=DECISION_POINTS,
                               aggregation_steps=10,
                               knapsack_capacity=KNAPSACK_CAPACITY)
        print("Start training for {} instances.\n".format(N_TRAIN_INSTANCES))
        qtable_policy.train(env=knapsack_problem, n_iterations=N_TRAIN_INSTANCES, evaluate_every_n=1000)
        print("\nFinished training!\n")
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print("Fitted Q-Table:")
        print(qtable_policy.qtable)
        print("Evaluate Q-table policy on {} instances".format(N_EVAL_INSTANCES))
        scores = evaluate_policy(env=knapsack_problem, policy=qtable_policy, n_instances=N_EVAL_INSTANCES)
        print("Mean total profit over all instances {:.3f}.".format(np.mean(scores)))
        print("Maximum total profit over all instances {:.3f}.".format(np.max(scores)))

    if POLICY == "PARAMETRIC":
        linreg_policy = LinearRegression(knapsack_capacity=KNAPSACK_CAPACITY,
                                         k_decision_points=DECISION_POINTS)
        print("Start training for {} instances.\n".format(N_TRAIN_INSTANCES))
        linreg_policy.train(env=knapsack_problem, n_iterations=N_TRAIN_INSTANCES, evaluate_every_n=1000)
        print("\nFinished training!\n")
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print("Fitted linear regression:")
        print(linreg_policy.a, linreg_policy.b)
        print("Evaluate Q-table policy on {} instances".format(N_EVAL_INSTANCES))
        scores = evaluate_policy(env=knapsack_problem, policy=linreg_policy, n_instances=N_EVAL_INSTANCES)
        print("Mean total profit over all instances {:.3f}.".format(np.mean(scores)))
        print("Maximum total profit over all instances {:.3f}.".format(np.max(scores)))
