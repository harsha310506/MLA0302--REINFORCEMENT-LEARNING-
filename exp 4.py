import numpy as np

# ---------------------------
# 1. Define the City Grid
# ---------------------------
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_MOVES = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

WAREHOUSE = (0, 0)  # starting point (not used directly in DP, just for reference)
DELIVERY_POINTS = [(4, 4), (2, 3)]   # multiple goal states
OBSTACLES = [(1, 1), (2, 2), (3, 3)] # no-fly zones

GAMMA = 0.9   # discount factor
STEP_REWARD = -1  # small penalty per move (encourages shortest path)
GOAL_REWARD = 10
OBSTACLE_REWARD = -5

# ---------------------------
# 2. Reward Function
# ---------------------------
def get_reward(state):
    if state in DELIVERY_POINTS:
        return GOAL_REWARD
    elif state in OBSTACLES:
        return OBSTACLE_REWARD
    else:
        return STEP_REWARD

# ---------------------------
# 3. Environment Transition
# ---------------------------
def is_valid(state):
    r, c = state
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def move(state, action):
    if state in DELIVERY_POINTS:
        return state  # terminal state, stays put
    dr, dc = ACTION_MOVES[action]
    new_state = (state[0] + dr, state[1] + dc)
    if not is_valid(new_state) or new_state in OBSTACLES:
        return state  # invalid move -> stay in place
    return new_state

# ---------------------------
# 4. Initialize Random Policy
# ---------------------------
def init_policy():
    policy = {}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            policy[(r, c)] = np.random.choice(ACTIONS)
    return policy

# ---------------------------
# 5. Policy Evaluation Step
# ---------------------------
def policy_evaluation(policy, V, theta=1e-4):
    while True:
        delta = 0
        for state in V.keys():
            if state in DELIVERY_POINTS:
                continue  # terminal state

            action = policy[state]
            next_state = move(state, action)
            reward = get_reward(next_state)
            new_value = reward + GAMMA * V[next_state]

            delta = max(delta, abs(new_value - V[state]))
            V[state] = new_value

        if delta < theta:
            break
    return V

# ---------------------------
# 6. Policy Improvement Step
# ---------------------------
def policy_improvement(policy, V):
    policy_stable = True

    for state in V.keys():
        if state in DELIVERY_POINTS:
            continue

        old_action = policy[state]

        # Try all actions, pick the one with highest value
        action_values = {}
        for action in ACTIONS:
            next_state = move(state, action)
            reward = get_reward(next_state)
            action_values[action] = reward + GAMMA * V[next_state]

        best_action = max(action_values, key=action_values.get)
        policy[state] = best_action

        if best_action != old_action:
            policy_stable = False

    return policy, policy_stable

# ---------------------------
# 7. Full Policy Iteration Algorithm
# ---------------------------
def policy_iteration():
    V = {(r, c): 0 for r in range(GRID_SIZE) for c in range(GRID_SIZE)}
    policy = init_policy()

    iteration = 0
    while True:
        iteration += 1
        V = policy_evaluation(policy, V)
        policy, stable = policy_improvement(policy, V)

        if stable:
            print(f"Optimal policy found after {iteration} iterations")
            break

    return policy, V

# ---------------------------
# 8. Run Policy Iteration
# ---------------------------
optimal_policy, optimal_values = policy_iteration()

# ---------------------------
# 9. Display Results
# ---------------------------
ARROWS = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}

print("\nOptimal Policy (Drone Route Directions):")
for r in range(GRID_SIZE):
    row_display = []
    for c in range(GRID_SIZE):
        state = (r, c)
        if state in DELIVERY_POINTS:
            row_display.append(" G ")   # Goal/delivery point
        elif state in OBSTACLES:
            row_display.append(" X ")   # Obstacle
        else:
            row_display.append(f" {ARROWS[optimal_policy[state]]} ")
    print("".join(row_display))

print("\nOptimal Value Function (Grid View):")
for r in range(GRID_SIZE):
    row_values = [f"{optimal_values[(r, c)]:6.2f}" for c in range(GRID_SIZE)]
    print(" | ".join(row_values))
