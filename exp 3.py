import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Define Pricing Environment
# ---------------------------
np.random.seed(42)

PRICES = [10, 15, 20, 25, 30]  # possible prices (arms)
N_ARMS = len(PRICES)
N_ROUNDS = 2000  # number of pricing decisions (customers)

# True (unknown to algorithm) purchase probability for each price
# Higher price -> lower probability of purchase (realistic assumption)
TRUE_CONVERSION_PROB = [0.65, 0.55, 0.40, 0.25, 0.12]

def get_reward(arm_index):
    """Simulate a customer's purchase decision and return revenue."""
    price = PRICES[arm_index]
    purchase = np.random.rand() < TRUE_CONVERSION_PROB[arm_index]
    return price if purchase else 0

# ---------------------------
# 2. Epsilon-Greedy Strategy
# ---------------------------
def epsilon_greedy(epsilon=0.1):
    counts = np.zeros(N_ARMS)
    values = np.zeros(N_ARMS)  # estimated average reward per arm
    rewards = []

    for t in range(N_ROUNDS):
        if np.random.rand() < epsilon:
            arm = np.random.randint(N_ARMS)  # explore
        else:
            arm = np.argmax(values)  # exploit

        reward = get_reward(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]  # incremental average
        rewards.append(reward)

    return rewards

# ---------------------------
# 3. UCB (Upper Confidence Bound) Strategy
# ---------------------------
def ucb(c=2):
    counts = np.zeros(N_ARMS)
    values = np.zeros(N_ARMS)
    rewards = []

    # Initial pass: try each arm once
    for arm in range(N_ARMS):
        reward = get_reward(arm)
        counts[arm] += 1
        values[arm] = reward
        rewards.append(reward)

    for t in range(N_ARMS, N_ROUNDS):
        ucb_scores = values + c * np.sqrt(np.log(t + 1) / counts)
        arm = np.argmax(ucb_scores)

        reward = get_reward(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)

    return rewards

# ---------------------------
# 4. Thompson Sampling Strategy
# ---------------------------
def thompson_sampling():
    # Model purchase (Bernoulli) with Beta distribution, scaled by price for revenue
    alpha = np.ones(N_ARMS)  # successes + 1
    beta = np.ones(N_ARMS)   # failures + 1
    rewards = []

    for t in range(N_ROUNDS):
        sampled_probs = np.random.beta(alpha, beta)
        # Expected revenue = sampled conversion probability * price
        expected_revenue = sampled_probs * np.array(PRICES)
        arm = np.argmax(expected_revenue)

        price = PRICES[arm]
        purchase = np.random.rand() < TRUE_CONVERSION_PROB[arm]
        reward = price if purchase else 0

        if purchase:
            alpha[arm] += 1
        else:
            beta[arm] += 1

        rewards.append(reward)

    return rewards

# ---------------------------
# 5. Run All Strategies
# ---------------------------
eg_rewards = epsilon_greedy(epsilon=0.1)
ucb_rewards = ucb(c=2)
ts_rewards = thompson_sampling()

# ---------------------------
# 6. Compare Results
# ---------------------------
print("Total Revenue Comparison over", N_ROUNDS, "pricing decisions:")
print(f"Epsilon-Greedy   : ${sum(eg_rewards):.2f}")
print(f"UCB              : ${sum(ucb_rewards):.2f}")
print(f"Thompson Sampling: ${sum(ts_rewards):.2f}")

# ---------------------------
# 7. Plot Cumulative Revenue
# ---------------------------
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(eg_rewards), label="Epsilon-Greedy")
plt.plot(np.cumsum(ucb_rewards), label="UCB")
plt.plot(np.cumsum(ts_rewards), label="Thompson Sampling")
plt.xlabel("Pricing Decision (Round)")
plt.ylabel("Cumulative Revenue ($)")
plt.title("Cumulative Revenue Comparison: Pricing Strategies")
plt.legend()
plt.grid(True)
plt.show()
