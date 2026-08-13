import random

# -----------------------------
# User Input
# -----------------------------
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))
episodes = int(input("Enter number of episodes: "))

alpha = float(input("Enter learning rate (alpha): "))
gamma = float(input("Enter discount factor (gamma): "))
epsilon = float(input("Enter exploration rate (epsilon): "))

# -----------------------------
# Create Grid
# -----------------------------
grid = [['.' for _ in range(cols)] for _ in range(rows)]

print("\nEnter Food Position")
food_r = int(input("Row: "))
food_c = int(input("Column: "))

print("\nEnter Ghost Position")
ghost_r = int(input("Row: "))
ghost_c = int(input("Column: "))

grid[food_r][food_c] = 'F'
grid[ghost_r][ghost_c] = 'G'

start = (0, 0)

actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# -----------------------------
# Q Table
# -----------------------------
Q = {}

for i in range(rows):
    for j in range(cols):
        for action in actions:
            Q[((i, j), action)] = 0

# -----------------------------
# Choose Action
# -----------------------------
def choose_action(state):

    if random.random() < epsilon:
        return random.choice(actions)

    values = [Q[(state, a)] for a in actions]
    best = max(values)

    best_actions = []

    for a in actions:
        if Q[(state, a)] == best:
            best_actions.append(a)

    return random.choice(best_actions)

# -----------------------------
# Move Agent
# -----------------------------
def move(state, action):

    r, c = state

    if action == "UP":
        r -= 1

    elif action == "DOWN":
        r += 1

    elif action == "LEFT":
        c -= 1

    elif action == "RIGHT":
        c += 1

    if r < 0 or r >= rows or c < 0 or c >= cols:
        return state, -5, False

    if (r, c) == (ghost_r, ghost_c):
        return (r, c), -50, True

    if (r, c) == (food_r, food_c):
        return (r, c), 100, True

    return (r, c), -1, False

# -----------------------------
# Training
# -----------------------------
print("\nTraining Started...\n")

for ep in range(episodes):

    state = start
    total_reward = 0

    for step in range(100):

        action = choose_action(state)

        next_state, reward, done = move(state, action)

        max_next = max(Q[(next_state, a)] for a in actions)

        old_q = Q[(state, action)]

        Q[(state, action)] = old_q + alpha * (
            reward + gamma * max_next - old_q
        )

        state = next_state
        total_reward += reward

        if done:
            break

    print("Episode", ep + 1, "Reward =", total_reward)

# -----------------------------
# Learned Policy
# -----------------------------
print("\nOptimal Policy\n")

for i in range(rows):

    for j in range(cols):

        if (i, j) == (food_r, food_c):
            print(" F ", end=" ")

        elif (i, j) == (ghost_r, ghost_c):
            print(" G ", end=" ")

        else:

            best = max(actions, key=lambda a: Q[((i, j), a)])

            if best == "UP":
                print(" ↑ ", end=" ")

            elif best == "DOWN":
                print(" ↓ ", end=" ")

            elif best == "LEFT":
                print(" ← ", end=" ")

            else:
                print(" → ", end=" ")

    print()

# -----------------------------
# Test Agent
# -----------------------------
print("\nAgent Simulation\n")

state = start

for step in range(20):

    action = max(actions, key=lambda a: Q[(state, a)])

    next_state, reward, done = move(state, action)

    print("Step", step + 1,
          "| State:", state,
          "| Action:", action,
          "| Reward:", reward)

    state = next_state

    if done:
        if state == (food_r, food_c):
            print("\nFood Collected! Agent Wins.")
        else:
            print("\nGhost Caught the Agent!")
        break
