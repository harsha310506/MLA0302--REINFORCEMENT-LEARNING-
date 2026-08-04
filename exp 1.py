"""
Autonomous Cleaning Robot - MDP Simulation
--------------------------------------------
A 5x5 grid world where a robot must clean dirt cells (+1 reward)
while avoiding obstacles (-1 penalty).

The problem is modeled as a Markov Decision Process (MDP):
    States      -> each (row, col) cell on the grid
    Actions     -> Up, Down, Left, Right
    Transition  -> deterministic move in the chosen direction
                   (staying in place if it would leave the grid)
    Reward      -> +1 for entering a dirt cell (cleaned after visit)
                   -1 for entering an obstacle cell
                   -0.04 "living cost" for every other move
                   (encourages the shortest / most efficient path)
    Discount    -> gamma = 0.9

We solve the MDP with Value Iteration to get an optimal policy,
then simulate the robot following that policy, re-planning every
time a dirt cell is cleaned (since the reward map changes).

For comparison, a purely random policy is also simulated so you can
see the difference in efficiency.

Run this file directly in Python IDLE (F5) - no external libraries required.
"""

import random

GRID_SIZE = 5
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

GAMMA = 0.9          # discount factor
LIVING_REWARD = -0.04  # small penalty per step to encourage efficiency
THETA = 1e-4          # convergence threshold for value iteration
MAX_ITERATIONS = 1000


class GridWorld:
    """Represents the 5x5 cleaning environment."""

    def __init__(self, dirt_cells, obstacle_cells, start=(0, 0)):
        self.size = GRID_SIZE
        self.start = start
        self.dirt = set(dirt_cells)
        self.obstacles = set(obstacle_cells)

    def in_bounds(self, state):
        r, c = state
        return 0 <= r < self.size and 0 <= c < self.size

    def reward(self, state):
        """Reward for ENTERING this state."""
        if state in self.obstacles:
            return -1.0
        if state in self.dirt:
            return 1.0
        return LIVING_REWARD

    def step(self, state, action):
        """Deterministic transition. Obstacles block movement (robot bounces back)."""
        dr, dc = ACTION_DELTA[action]
        next_state = (state[0] + dr, state[1] + dc)
        if not self.in_bounds(next_state):
            next_state = state  # bump into wall, stay in place
        return next_state

    def all_states(self):
        return [(r, c) for r in range(self.size) for c in range(self.size)]


def value_iteration(env):
    """Computes the optimal value function and policy for the current grid."""
    V = {s: 0.0 for s in env.all_states()}

    for _ in range(MAX_ITERATIONS):
        delta = 0.0
        new_V = V.copy()
        for s in env.all_states():
            if s in env.obstacles:
                continue  # terminal-ish: robot avoids planning through here
            best_value = float("-inf")
            for a in ACTIONS:
                s_next = env.step(s, a)
                r = env.reward(s_next)
                value = r + GAMMA * V[s_next]
                best_value = max(best_value, value)
            new_V[s] = best_value
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < THETA:
            break

    # Extract greedy policy from V
    policy = {}
    for s in env.all_states():
        if s in env.obstacles:
            policy[s] = None
            continue
        best_action, best_value = None, float("-inf")
        for a in ACTIONS:
            s_next = env.step(s, a)
            r = env.reward(s_next)
            value = r + GAMMA * V[s_next]
            if value > best_value:
                best_value = value
                best_action = a
        policy[s] = best_action

    return V, policy


def print_grid(env, robot_pos=None):
    """Pretty-prints the current grid state."""
    for r in range(env.size):
        row_str = ""
        for c in range(env.size):
            cell = (r, c)
            if robot_pos == cell:
                row_str += " R "
            elif cell in env.obstacles:
                row_str += " X "
            elif cell in env.dirt:
                row_str += " D "
            else:
                row_str += " . "
        print(row_str)
    print()


def simulate_optimal_policy(env, max_steps=100):
    """
    Runs the robot using Value Iteration, re-planning after every
    dirt cell is cleaned (since cleaning changes the reward map).
    """
    print("=" * 50)
    print("SIMULATING OPTIMAL (VALUE ITERATION) POLICY")
    print("=" * 50)

    pos = env.start
    total_reward = 0.0
    steps = 0

    print("Initial grid:")
    print_grid(env, pos)

    while env.dirt and steps < max_steps:
        _, policy = value_iteration(env)
        action = policy[pos]
        if action is None:
            break
        next_pos = env.step(pos, action)
        r = env.reward(next_pos)
        total_reward += r

        if next_pos in env.dirt:
            print(f"Step {steps+1}: {pos} --{action}--> {next_pos}  (cleaned dirt! reward {r:+.2f})")
            env.dirt.remove(next_pos)
        elif next_pos in env.obstacles:
            print(f"Step {steps+1}: {pos} --{action}--> {next_pos}  (hit obstacle! reward {r:+.2f})")
        else:
            print(f"Step {steps+1}: {pos} --{action}--> {next_pos}  (reward {r:+.2f})")

        pos = next_pos
        steps += 1

    print("\nFinal grid:")
    print_grid(env, pos)
    print(f"Finished in {steps} steps with total reward = {total_reward:.2f}")
    print(f"Remaining dirt cells: {len(env.dirt)}\n")
    return steps, total_reward


def simulate_random_policy(env, max_steps=100):
    """Runs the robot picking random actions, for comparison."""
    print("=" * 50)
    print("SIMULATING RANDOM POLICY (baseline)")
    print("=" * 50)

    pos = env.start
    total_reward = 0.0
    steps = 0

    while env.dirt and steps < max_steps:
        action = random.choice(ACTIONS)
        next_pos = env.step(pos, action)
        r = env.reward(next_pos)
        total_reward += r

        if next_pos in env.dirt:
            env.dirt.remove(next_pos)

        pos = next_pos
        steps += 1

    print(f"Finished in {steps} steps with total reward = {total_reward:.2f}")
    print(f"Remaining dirt cells: {len(env.dirt)}\n")
    return steps, total_reward


if __name__ == "__main__":
    random.seed(42)

    # Define the grid layout
    dirt_cells = [(0, 3), (1, 1), (2, 4), (3, 0), (4, 2)]
    obstacle_cells = [(1, 3), (2, 1), (3, 3)]

    # --- Run optimal policy ---
    env1 = GridWorld(dirt_cells, obstacle_cells, start=(0, 0))
    opt_steps, opt_reward = simulate_optimal_policy(env1)

    # --- Run random policy on a fresh copy of the same layout ---
    env2 = GridWorld(dirt_cells, obstacle_cells, start=(0, 0))
    rand_steps, rand_reward = simulate_random_policy(env2)

    # --- Comparison summary ---
    print("=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"{'Policy':<12}{'Steps':<10}{'Total Reward':<15}")
    print(f"{'Optimal':<12}{opt_steps:<10}{opt_reward:<15.2f}")
    print(f"{'Random':<12}{rand_steps:<10}{rand_reward:<15.2f}")
