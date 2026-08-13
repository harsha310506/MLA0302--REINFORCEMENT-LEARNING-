import random

# --------------------------------
# User Input
# --------------------------------

num_representatives = int(input("Enter number of representatives: "))
num_call_types = int(input("Enter number of call types: "))
episodes = int(input("Enter number of training episodes: "))

epsilon = float(input("Enter exploration rate: "))


# --------------------------------
# Create Environment
# --------------------------------

representatives = []

for i in range(num_representatives):
    representatives.append("REP_" + str(i+1))


calls = []

for i in range(num_call_types):
    calls.append("CALL_TYPE_" + str(i+1))


# Random handling time dataset
handling_time = {}

for call in calls:
    for rep in representatives:
        handling_time[(call, rep)] = random.randint(2,10)


print("\nCall Handling Time Dataset")

for key,value in handling_time.items():
    print(key, "=", value,"minutes")


# --------------------------------
# Initialize Policy
# --------------------------------

policy = {}

Q = {}

Returns = {}


for call in calls:

    for rep in representatives:

        Q[(call,rep)] = 0
        Returns[(call,rep)] = []


    policy[call] = random.choice(representatives)



# --------------------------------
# Choose Action
# --------------------------------

def choose_action(call):

    if random.random() < epsilon:
        return random.choice(representatives)

    best = -999

    action = None

    for rep in representatives:

        if Q[(call,rep)] > best:
            best = Q[(call,rep)]
            action = rep

    return action



# --------------------------------
# Monte Carlo Policy Control
# --------------------------------

for episode in range(episodes):

    episode_data = []

    for call in calls:

        rep = choose_action(call)

        time = handling_time[(call,rep)]

        reward = -time

        episode_data.append(
            (call,rep,reward)
        )


    visited = set()


    for call,rep,reward in episode_data:

        if (call,rep) not in visited:

            visited.add((call,rep))


            Returns[(call,rep)].append(reward)


            Q[(call,rep)] = sum(
                Returns[(call,rep)]
            ) / len(Returns[(call,rep)])


    # Improve policy

    for call in calls:

        best_rep = max(
            representatives,
            key=lambda r: Q[(call,r)]
        )

        policy[call] = best_rep



    if (episode+1)%10 == 0:

        print(
            "Episode",
            episode+1,
            "completed"
        )


# --------------------------------
# Final Optimal Policy
# --------------------------------

print("\nOptimal Call Assignment Policy")

for call in calls:

    print(
        call,
        "---->",
        policy[call]
    )


# --------------------------------
# Calculate Average Handling Time
# --------------------------------

total_time = 0


for call in calls:

    rep = policy[call]

    time = handling_time[(call,rep)]

    total_time += time


average_time = total_time / len(calls)


print("\nAverage Call Handling Time:",
      round(average_time,2),
      "minutes")
