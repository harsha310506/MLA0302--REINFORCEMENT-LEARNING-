import random

# ===========================
# DATASET (House Grid)
# ===========================
# S = Start
# D = Dirt
# X = Obstacle
# . = Empty

house = [
    ['S', '.', '.', 'D', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', '.', 'D', '.', '.'],
    ['.', 'X', '.', '.', 'D'],
    ['.', '.', '.', '.', '.']
]

ROWS = len(house)
COLS = len(house[0])

actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# ===========================
# USER INPUT
# ===========================
episodes = int(input("Enter number of episodes: "))
alpha = float(input("Enter learning rate (alpha): "))
gamma = float(input("Enter discount factor (gamma): "))
epsilon = float(input("Enter exploration rate (epsilon): "))

max_steps = 100

# ===========================
# Q TABLE
# ===========================
Q = {}

for i in range(ROWS):
    for j in range(COLS):
        for action in actions:
            Q[((i, j), action)] = 0.0

# ===========================
# EPSILON GREEDY
# ===========================
def choose_action(state):

    if random.random() < epsilon:
        return random.choice(actions)

    best = max(Q[(state, a)] for a in actions)

    best_actions = [a for a in actions if Q[(state, a)] == best]

    return random.choice(best_actions)

# ===========================
# MOVE FUNCTION
# ===========================
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

    if r < 0 or r >= ROWS or c < 0 or c >= COLS:
        return state, -5

    cell = house[r][c]

    if cell == 'X':
        return state, -10

    elif cell == 'D':
        return (r, c), 10

    else:
        return (r, c), -1

# ===========================
# SARSA TRAINING
# ===========================
print("\nTraining Started...\n")

for episode in range(episodes):

    state = (0, 0)

    action = choose_action(state)

    total_reward = 0

    for step in range(max_steps):

        next_state, reward = move(state, action)

        next_action = choose_action(next_state)

        old_q = Q[(state, action)]

        next_q = Q[(next_state, next_action)]

        Q[(state, action)] = old_q + alpha * (
            reward + gamma * next_q - old_q
        )

        state = next_state
        action = next_action

        total_reward += reward

    if (episode + 1) % 50 == 0 or episode == episodes - 1:
        print("Episode:", episode + 1,
              "Reward:", total_reward)

print("\nTraining Completed!")

# ===========================
# OPTIMAL POLICY
# ===========================
print("\nOptimal Cleaning Policy\n")

for i in range(ROWS):

    for j in range(COLS):

        if house[i][j] == 'X':
            print(" X ", end=" ")

        else:

            best_action = max(actions,
                              key=lambda a: Q[((i, j), a)])

            if best_action == "UP":
                print(" ↑ ", end=" ")

            elif best_action == "DOWN":
                print(" ↓ ", end=" ")

            elif best_action == "LEFT":
                print(" ← ", end=" ")

            else:
                print(" → ", end=" ")

    print()

# ===========================
# ROBOT SIMULATION
# ===========================
print("\nRobot Simulation\n")

state = (0, 0)

for step in range(20):

    action = max(actions, key=lambda a: Q[(state, a)])

    next_state, reward = move(state, action)

    print("Step:", step + 1,
          "| State:", state,
          "| Action:", action,
          "| Reward:", reward)

    state = next_state
