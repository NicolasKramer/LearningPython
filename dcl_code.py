import numpy as np
import random
import matplotlib.pyplot as plt


def cluster_points(points, mu):
    clusters = {}
    for x in points:
        best_mu_key = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[best_mu_key].append(x)
        except KeyError:
            clusters[best_mu_key] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(x, k):
    # Initialize to K random centers
    oldmu = random.sample(x, k)
    mu = random.sample(x, k)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(x, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


def init_board(n):
    ar = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(n)])
    return ar


def init_board_gauss(total_points, k):
    n = float(total_points)/k
    ar = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        ar.extend(x)
    ar = np.array(ar)[:total_points]
    return ar

X = init_board_gauss(30, 3)
# X = init_board(200)
xs = []
ys = []

for i in X:
    xs.append(i[0])
    ys.append(i[1])

plt.plot(xs, ys, 'ro')
plt.axis([-1, 1, -1, 1])
plt.show()
