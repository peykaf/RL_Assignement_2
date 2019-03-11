import gym
import numpy as np
import matplotlib.pyplot as plt


saveFigures = False
plot_graph_ = True


class TaxiEnv:

    def __init__(self, method, alpha, temperature, env, gamma):
        self.env = env
        self.method = method
        self.positions = 25
        self.num_ps_location = 5
        self.num_ps_destination = 4
        self.num_states = self.positions * self.num_ps_location * self.num_ps_destination
        self.num_actions = 6
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        # initialize q table
        self.q = np.zeros(shape=(self.num_states, self.num_actions))
        # initialize policy
        self.pi = np.zeros_like(self.q)

    def interact(self):
        state = self.env.reset()
        delivered = False
        r_sum = 0
        steps = 0
        while not delivered:
            steps += 1
            action, probs = self._take_action(state)
            next_state, reward, delivered, info = self.env.step(action)
            if self.method == 'sarsa':
                next_action = self._take_action(next_state)[0]
                self.q[state][action] += self.alpha * \
                                         (reward + self.gamma * self.q[next_state][next_action] - self.q[state][action])
            r_sum += reward
            state = next_state
        return r_sum

    def _take_action(self, state):
        self.pi[state] = np.exp(self.q[state] / self.temperature)
        prob = np.true_divide(self.pi[state], sum(self.pi[state]))
        a = np.random.choice(np.arange(6), p=prob)
        return a, prob

    def run_optimal_policy(self):
        state = self.env.reset()
        delivered = False
        r_sum = 0
        steps = 0
        while delivered == False:
            steps += 1
            action = np.random.choice(np.where(self.q[state] == max(self.q[state]))[0])
            next_state, reward, delivered, info = self.env.step(action)
            r_sum += reward
            state = next_state

            if delivered:
                print(end="")

        return r_sum

    def train(self, print_trace=False):

        runs = 10
        segments = 100
        episodes = 10

        train_rewards = []
        test_rewards = []

        reward_train_episodes = []
        reward_test_episode = []

        for run in range(runs):
            self.q = np.zeros(shape=(self.num_states, self.num_actions))
            for segment in range(segments):
                train_rewards = []
                for episode in range(episodes + 1):
                    if episode < episodes:
                        reward_sum = self.interact()
                        train_rewards.append(reward_sum)
                    else:
                        test_reward = self.run_optimal_policy()
                        test_rewards.append(test_reward)
                    if print_trace:
                        if episode == episodes and segment % 10 == 0:
                            print('alpha {}, tau {} | Run {}, Segment {}'.
                                  format(self.alpha, self.temperature, run, segment))
                    if episode == episodes and segment == segments - 1:  # last segment, last episode
                        reward_train_episodes.append(np.mean(train_rewards))
                        reward_test_episode.append(test_rewards[-1])

        train_per_run = np.mean(reward_train_episodes)  # for the first bullet in the question
        test_per_run = np.mean(reward_test_episode)     # for the second bullet in the question

        return train_per_run, test_per_run


def main():
    env = gym.make('Taxi-v2')
    temperatures = [0.5, 1, 1.5]
    learning_rates = [0.3, 0.5, 0.8]

    sarsa_train_rewards = np.zeros(shape=(len(temperatures), len(learning_rates)))
    sarsa_test_rewards = np.zeros_like(sarsa_train_rewards)

    for temp in temperatures:
        for learning_rate in learning_rates:
            sarsa_agent = TaxiEnv('sarsa', learning_rate, temp, env, gamma=0.9)
            train_rewards, test_rewards = sarsa_agent.train(print_trace=True)
            sarsa_train_rewards[temperatures.index(temp)][learning_rates.index(learning_rate)] = train_rewards
            sarsa_test_rewards[temperatures.index(temp)][learning_rates.index(learning_rate)] = test_rewards

    print(sarsa_train_rewards)
    print(sarsa_test_rewards)

    plot_graph(learning_rates,
               sarsa_train_rewards,
               temperatures,
               'learning rate',
               'Av. return during training sarsa',
               'av_return_l_rate_training_sarsa.png')


def plot_graph(x_value, y_values, legend_names, x_axis_title, y_axis_title, filename_to_save):
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(y_values)):
        ax.plot(x_value,
                y_values[i],
                label=legend_names[i])
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)

    plt.legend(loc='upper left')
    if saveFigures:
        plt.savefig(filename_to_save)
    if plot_graph_:
        plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
