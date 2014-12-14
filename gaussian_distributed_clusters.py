__author__ = 'nicolas'
import numpy as np
import random


def init_board_gauss(n, k):
    n = float(n)/k
    ar = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05, 0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a, b])
        ar.extend(x)
    ar = np.array(ar)[:n]
    return ar