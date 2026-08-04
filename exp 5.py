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

PICKUP_POINTS = [(0, 4), (4, 0)]     # multiple pickup locations (goal states)
OBSTACLES = [(1, 2), (2, 2), (3, 2)] # blocked roads / heavy traffic

GAMMA = 0.9
STEP_REWARD = -1       # small penalty per move -> encourages fastest pickup
GOAL_REWARD = 10
OBSTACLE_REWARD = -5

# ---------------------------
# 2. Reward Function
# ---------------------------
def get_reward(state):
    if state in PICKUP_POINTS:
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
    if state in PICKUP_POINTS:
        return state  # terminal state, stays put
    dr, dc = ACTION_MOVES[action]
    new_state = (state[0] + dr, state[1] + dc)
    if not is_valid(new_state) or new_state in OBSTACLES:
        return state  # blocked move -> stay in place
    return new_state

# ---------------------------
# 4. Value Iteration Algorithm
# ---------------------------
def value_iteration(theta=1e-4, max_iterations=1000):
    V = {(r, c): 0 for r in range(GRID_SIZE) for c in range(GRID_SIZE)}

    for iteration in range(max_iterations):
        delta = 0

        for state in V.keys():
            if state in PICKUP_POINTS:
                continue  # terminal state stays 0 / fixed

            # Try all actions, take the BEST one (max)
            action_values = []
            for action in ACTIONS:
                next_state = move(state, action)
                reward = get_reward(next_state)
                action_values.append(reward + GAMMA * V[next_state])

            best_value = max(action_values)
            delta = max(delta, abs(best_value - V[state]))
            V[state] = best_value

        if delta < theta:
            print(f"Value Iteration converged after {iteration+1} iterations")
            break

    return V

# ---------------------------
# 5. Extract Optimal Policy from Value Function
# ---------------------------
def extract_policy(V):
    policy = {}

    for state in V.keys():
        if state in PICKUP_POINTS:
            policy[state] = None  # no action needed at goal
            continue

        action_values = {}
        for action in ACTIONS:
            next_state = move(state, action)
            reward = get_reward(next_state)
            action_values[action] = reward + GAMMA * V[next_state]

        best_action = max(action_values, key=action_values.get)
        policy[state] = best_action

    return policy

# ---------------------------
# 6. Run Value Iteration
# ---------------------------
optimal_values = value_iteration()
optimal_policy = extract_policy(optimal_values)

# ---------------------------
# 7. Display Results
# ---------------------------
ARROWS = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}

print("\nOptimal Dispatch Policy (Taxi Route Directions):")
for r in range(GRID_SIZE):
    row_display = []
    for c in range(GRID_SIZE):
        state = (r, c)
        if state in PICKUP_POINTS:
            row_display.append(" P ")   # Pickup point
        elif state in OBSTACLES:
            row_display.append(" X ")   # Obstacle
        else:
            row_display.append(f" {ARROWS[optimal_policy[state]]} ")
    print("".join(row_display))

print("\nOptimal Value Function (Grid View):")
for r in range(GRID_SIZE):
    row_values = [f"{optimal_values[(r, c)]:6.2f}" for c in range(GRID_SIZE)]
    print(" | ".join(row_values))
