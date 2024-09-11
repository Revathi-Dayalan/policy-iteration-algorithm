# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.


## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

## POLICY ITERATION ALGORITHM
Include the steps involved in policy iteration algorithm

## POLICY IMPROVEMENT FUNCTION
### Name : ADHITHYARAM D
### Register Number : 212222230008
```PYTHON
def policy_improvement(V, P, gamma=0.9):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_policy = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_policy
```
## POLICY ITERATION FUNCTION
### Name : ADHITHYARAM D
### Register Number : 212222230008
```python
def policy_iteration(P, gamma=0.9, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
    while True:
        old_pi = {s: pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)
        if old_pi == {s: pi(s) for s in range(len(P))}:
            break
    return V, pi
```
## OUTPUT:

## optimal policy 

<img src="https://github.com/user-attachments/assets/6fe4e64e-c272-4ae5-abbf-84e03c228eef" width=50%>

## optimal value function

<img src="https://github.com/user-attachments/assets/7fc48bde-f516-4a37-bdf5-ee1758627a11" width=50%>

## success rate for the optimal policy

<img src="https://github.com/user-attachments/assets/2f47ec4a-7a60-49f9-81a9-7c043c49b580" width=50%>


## RESULT:

Thus, Python program is developed to find the optimal policy for the given MDP using the policy iteration algorithm.

