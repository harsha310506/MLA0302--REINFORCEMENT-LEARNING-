import random

# -------------------------------
# User Input
# -------------------------------
prices = list(map(float, input("Enter stock prices separated by space:\n").split()))

episodes = int(input("Enter number of episodes: "))
alpha = float(input("Enter learning rate (alpha): "))
gamma = float(input("Enter discount factor (gamma): "))
epsilon = float(input("Enter epsilon: "))

actions = ["BUY", "SELL", "HOLD"]

# Q Table
Q = {}

for i in range(len(prices)):
    for a in actions:
        Q[(i, a)] = 0.0


def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)

    values = [Q[(state, a)] for a in actions]
    best = max(values)

    best_actions = [a for a in actions if Q[(state, a)] == best]

    return random.choice(best_actions)


# -------------------------------
# Training
# -------------------------------
for episode in range(episodes):

    holding = False
    buy_price = 0
    total_reward = 0

    for state in range(len(prices) - 1):

        action = choose_action(state)

        reward = 0

        # BUY
        if action == "BUY" and not holding:
            holding = True
            buy_price = prices[state]

        # SELL
        elif action == "SELL" and holding:
            reward = prices[state] - buy_price
            holding = False

        # HOLD
        else:
            reward = -0.1

        next_state = state + 1

        next_action = choose_action(next_state)

        current_q = Q[(state, action)]
        next_q = Q[(next_state, next_action)]

        # Double DQN-style update (simplified)
        target = reward + gamma * next_q

        Q[(state, action)] = current_q + alpha * (target - current_q)

        total_reward += reward

    print("Episode", episode + 1, "Reward =", round(total_reward, 2))

# -------------------------------
# Testing
# -------------------------------
print("\n===== Learned Trading Strategy =====")

holding = False
buy_price = 0
profit = 0

for state in range(len(prices) - 1):

    action = max(actions, key=lambda a: Q[(state, a)])

    print("Day", state + 1,
          "| Price =", prices[state],
          "| Action =", action)

    if action == "BUY" and not holding:
        holding = True
        buy_price = prices[state]

    elif action == "SELL" and holding:
        gain = prices[state] - buy_price
        profit += gain
        print(" Sold | Profit =", round(gain, 2))
        holding = False

print("\nTotal Profit =", round(profit, 2))
