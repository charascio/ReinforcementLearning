import numpy as np

np.random.seed(0)

P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]

gamma = 0.5

def compute_return(start_index, chain, gamma):
    # G = 0
    # for i in reversed(range(start_index, len(chain))):
    #     G = gamma * G + rewards[chain[i]-1]
    # return G
    G = 0
    for i in range(start_index, len(chain)):
        G += rewards[chain[i]-1] * np.power(gamma, i)
    return G

chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print(G)

def compute(P, rewards, gamma, states_num):
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num)-gamma*P), rewards)
    return value

V = compute(P, rewards, gamma, 6)
print(V)

S = ["s1", "s2", "s3", "s4", "s5"]
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]
P = {
    "s1-保持s1-s1":1.0, "s1-前往s2-s2":1.0,
    "s2-前往s1-s1":1.0, "s2-前往s3-s3":1.0,
    "s3-前往s4-s4":1.0, "s3-前往s5-s5":1.0,
    "s4-前往s5-s5":1.0, "s4-概率前往-s2":0.2,
    "s4-概率前往-s3":0.4, "s4-概率前往-s4":0.4
}
R = {
    "s1-保持s1":-1, "s1-前往s2":0,
    "s2-前往s1":-1, "s2-前往s3":-2,
    "s3-前往s4":-2, "s3-前往s5":0,
    "s4-前往s5":10, "s4-概率前往":1,
}
gamma = 0.5
MDP = (S, A, P, R, gamma)

Pi_1 = {
    "s1-保持s1":0.5, "s1-前往s2":0.5,
    "s2-前往s1":0.5, "s2-前往s3":0.5,
    "s3-前往s4":0.5, "s3-前往s5":0.5,
    "s4-前往s5":0.5, "s4-概率前往":0.5,
}
Pi_2 = {
    "s1-保持s1":0.6, "s1-前往s2":0.4,
    "s2-前往s1":0.3, "s2-前往s3":0.7,
    "s3-前往s4":0.5, "s3-前往s5":0.5,
    "s4-前往s5":0.1, "s4-概率前往":0.9,
}

def join(str1, str2):
    return str1 + '-' +str2

P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
print(V)

def sample(MDP, Pi, timestep_max, number):
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]
        while s != "s5" and timestep < timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes

episodes = sample(MDP, Pi_1, 20, 5)
print(episodes)

def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode)-1, -1, -1):
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]

timestep_max = 20
episodes = sample(MDP, Pi_1, timestep_max, 1000)
gamma = 0.5
V = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0}
N = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0}
MC(episodes, V, N, gamma)
print(V)

def occupancy(episodes, s, a, timestep_max, gamma):
    rho = 0
    total_times = np.zeros(timestep_max)
    occur_times = np.zeros(timestep_max)
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1-gamma) * rho

gamma = 0.5
timestep_max = 20
episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
rho1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
print(rho1, rho2)