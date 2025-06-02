import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)

        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k):
        # 返回是否获奖，但是获奖概率仍然需要多次探索，即可能存在随机性
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        
np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print(f"获奖概率最大的是{bandit_10_arm.best_idx}号，获奖概率是{bandit_10_arm.best_prob}")

class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        return NotImplementedError
    
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k]+1) * (r-self.estimates[k])
        return k
    
def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regrets")
    plt.title(f"{solvers[0].bandit.K}-armed bandit")
    plt.legend()
    plt.show()

np.random.seed(1)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, e) for e in epsilons]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1. / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k]+1) * (r-self.estimates[k])
        return k
    
np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
plot_results([decaying_epsilon_greedy_solver], ["decaying_epsilon_greedy"])


class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2*(self.counts+1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k]+1) * (r-self.estimates[k])
        return k
    
np.random.seed(1)
coef = 1
UCB_Solver = UCB(bandit_10_arm, coef)
UCB_Solver.run(5000)
plot_results([UCB_Solver], ["UCB_Solver"])

class ThompsonSamp(Solver):
    def __init__(self, bandit):
        super(ThompsonSamp, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)
        self._a[k] += r
        self._b[k] += (1-r)
        return k

np.random.seed(1)
thompsonsame_solver = ThompsonSamp(bandit_10_arm)
thompsonsame_solver.run(5000)
plot_results([thompsonsame_solver], ["thompsonsame_solver"])


    

