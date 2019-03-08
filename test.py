import numpy as np


x = [1.0, 2, 6, -2.5, 0]

prob_t = [0,0,0,0,0]       #initialise
for a in range(5):
    prob_t[a] = np.exp(x[a] / 3)  #calculate numerators

#numpy matrix element-wise division for denominator (sum of numerators)
prob_t = np.true_divide(prob_t,sum(prob_t))
action = []
for i in range(20):
    action.append(np.random.choice(np.arange(5), p=prob_t))

print(prob_t)
print(action)