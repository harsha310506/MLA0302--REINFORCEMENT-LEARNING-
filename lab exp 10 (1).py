import random

episodes = int(input("Enter Number of Episodes: "))
investment = float(input("Enter Initial Investment Amount: "))

print("\nInvestment Actions")
print("1. Buy")
print("2. Hold")
print("3. Sell")

policy = [0.33, 0.34, 0.33]

total_reward = 0

for episode in range(episodes):

    print("\nEpisode", episode + 1)

    action = random.choices(
        ["Buy", "Hold", "Sell"],
        weights=policy
    )[0]

    market = random.choice(["Up", "Down"])

    print("Market :", market)
    print("Action :", action)

    if market == "Up" and action == "Buy":
        reward = 10

    elif market == "Down" and action == "Sell":
        reward = 8

    elif action == "Hold":
        reward = 5

    else:
        reward = -5

    total_reward += reward

    if reward > 0:

        if action == "Buy":
            policy[0] += 0.05

        elif action == "Hold":
            policy[1] += 0.05

        else:
            policy[2] += 0.05

    total = sum(policy)

    policy[0] /= total
    policy[1] /= total
    policy[2] /= total

    print("Reward :", reward)
    print("Updated Policy :", [round(x,2) for x in policy])

average = total_reward / episodes

print("\n---------------------------")
print("Training Completed")
print("---------------------------")
print("Total Reward :", total_reward)
print("Average Reward :", round(average,2))

print("\nFinal Policy")

print("Buy  Probability :", round(policy[0],2))
print("Hold Probability :", round(policy[1],2))
print("Sell Probability :", round(policy[2],2))

if average >= 8:
    print("\nInvestment Policy : Excellent")

elif average >= 5:
    print("\nInvestment Policy : Good")

else:
    print("\nInvestment Policy : Needs Improvement")
