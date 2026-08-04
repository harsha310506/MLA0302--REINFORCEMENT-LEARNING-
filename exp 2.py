import numpy as np

# ---------------------------
# 1. Define the Warehouse Grid
# ---------------------------
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_MOVES = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# Define special cells
ITEM_LOCATION = (1, 3)      # picking an item here
GOAL_LOCATION = (4, 4)      # goal / drop-off point
OBSTACLES = [(2, 2), (3, 1), (0, 4)]  # obstacle cells

# ---------------------------
# 2. Reward Function
# ---------------------------
def get_reward(state):
    if state == GOAL_LOCATION:
        return 5
    elif state == ITEM_LOCATION:
        return 2
    elif state in OBSTACLES:
        return -2
    else:
        return -0.1  # small penalty to encourage efficiency

# ---------------------------
# 3. Environment Transition
# ---------------------------
def is_valid(state):
    r, c = state
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def move(state, action):
    dr, dc = ACTION_MOVES[action]
    new_state = (state[0] + dr, state[1] + dc)
    if not is_valid(new_state):
        return state  # stay in place if move goes off-grid
    return new_state

# ---------------------------
# 4. Define a Simple Policy
# Here: equal probability (random policy) for all 4 actions
# ---------------------------
def get_policy():
    policy = {}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            policy[(r, c)] = {a: 0.25 for a in ACTIONS}  # equal probability
    return policy

# ---------------------------
# 5. Policy Evaluation Algorithm
# ---------------------------
def policy_evaluation(policy, gamma=0.9, theta=1e-4, max_iterations=1000):
    # Initialize V(s) = 0 for all states
    V = {(r, c): 0 for r in range(GRID_SIZE) for c in range(GRID_SIZE)}

    for iteration in range(max_iterations):
        delta = 0
        new_V = V.copy()

        for state in V.keys():
            if state == GOAL_LOCATION:
                continue  # terminal state, value stays 0 (or fixed)

            v = 0
            for action, action_prob in policy[state].items():
                next_state = move(state, action)
                reward = get_reward(next_state)
                v += action_prob * (reward + gamma * V[next_state])

            new_V[state] = v
            delta = max(delta, abs(v - V[state]))

        V = new_V

        if delta < theta:  # convergence check
            print(f"Converged after {iteration+1} iterations")
            break

    return V

# ---------------------------
# 6. Run Policy Evaluation
# ---------------------------
policy = get_policy()
value_function = policy_evaluation(policy)

# ---------------------------
# 7. Display Value Function as Grid
# ---------------------------
print("\nValue Function (Grid View):")
for r in range(GRID_SIZE):
    row_values = [f"{value_function[(r, c)]:6.2f}" for c in range(GRID_SIZE)]
    print(" | ".join(row_values))
