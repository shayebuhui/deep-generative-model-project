import numpy as np
n = 10000
x = np.random.randn(n, 1) * np.sqrt(0.5)+2
y = np.random.randn(n, 1)-2
z = np.random.binomial(1, 0.5, [n, 1])
data = x * z + (1 - z) * y
np.save('./data.npy', data.astype('f'))