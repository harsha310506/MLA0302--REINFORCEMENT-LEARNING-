# Dynamic Programming - Policy Iteration for GridWorld

# -----------------------------
# User Input
# -----------------------------
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

goal_row = int(input("Enter Goal Row: "))
goal_col = int(input("Enter Goal Column: "))

num_obstacles = int(input("Enter number of obstacles: "))

obstacles = []

for i in range(num_obstacles):
    print("\nObstacle", i + 1)
    r = int(input("Row: "))
    c = int(input("Column: "))
    obstacles.append((r, c))

gamma = float(input("Enter Discount Factor (0-1): "))

# -----------------------------
# Actions
# -----------------------------
actions = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

# -----------------------------
# Initialize Value Function
# -----------------------------
V = {}

for i in range(rows):
    for j in range(cols):
        V[(i, j)] = 0

policy = {}

for i in range(rows):
    for j in range(cols):
        policy[(i, j)] = "UP"

goal = (goal_row, goal_col)

# -----------------------------
# Reward Function
# -----------------------------
def reward(state):

    if state == goal:
        return 100

    if state in obstacles:
        return -100

    return -1

# -----------------------------
# Next State
# -----------------------------
def next_state(state, action):

    r, c = state

    dr, dc = actions[action]

    nr = r + dr
    nc = c + dc

    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        return state

    if (nr, nc) in obstacles:
        return state

    return (nr, nc)

# -----------------------------
# Policy Iteration
# -----------------------------
stable = False

while not stable:

    # Policy Evaluation
    for _ in range(100):

        new_V = V.copy()

        for i in range(rows):
            for j in range(cols):

                state = (i, j)

                if state == goal:
                    continue

                if state in obstacles:
                    continue

                action = policy[state]

                ns = next_state(state, action)

                new_V[state] = reward(ns) + gamma * V[ns]

        V = new_V

    # Policy Improvement
    stable = True

    for i in range(rows):
        for j in range(cols):

            state = (i, j)

            if state == goal or state in obstacles:
                continue

            old_action = policy[state]

            best_action = old_action
            best_value = -999999

            for action in actions:

                ns = next_state(state, action)

                value = reward(ns) + gamma * V[ns]

                if value > best_value:
                    best_value = value
                    best_action = action

            policy[state] = best_action

            if best_action != old_action:
                stable = False

# -----------------------------
# Display Value Function
# -----------------------------
print("\nState Values\n")

for i in range(rows):
    for j in range(cols):

        state = (i, j)

        if state == goal:
            print(" G ", end="\t")

        elif state in obstacles:
            print(" X ", end="\t")

        else:
            print(round(V[state], 1), end="\t")

    print()

# -----------------------------
# Display Optimal Policy
# -----------------------------
print("\nOptimal Policy\n")

symbols = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→"
}

for i in range(rows):
    for j in range(cols):

        state = (i, j)

        if state == goal:
            print(" G ", end=" ")

        elif state in obstacles:
            print(" X ", end=" ")

        else:
            print(symbols[policy[state]], end=" ")

    print()

# -----------------------------
# Simulate Agent
# -----------------------------
print("\nAgent Path\n")

state = (0, 0)

steps = 0

while state != goal and steps < rows * cols * 2:

    print("Current State:", state)

    action = policy[state]

    print("Action:", action)

    state = next_state(state, action)

    steps += 1

print("Goal Reached:", state)
print("Total Steps:", steps)
