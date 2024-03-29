import numpy as np
import random
import matplotlib.pyplot as plt


def cluster_points(X, mu):
    # Returns a dictionary with key for cluster index and value a list of point sub-lists
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu


def has_converged(mu, oldmu):
    # returns a True or False. Can be used to stop the iteration loop
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    # Returns mu as a tuple for each k. Returns clusters as the usual dictionary
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X


def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

N = 200
k = 5
X = init_board_gauss(N, k)

mu, clusters = find_centers(X,k)

# plt.figure(0)
#
# for i in X:
#     xs = []
#     ys = []
#     colors = [0.9,  0.1,  0.1]
#     xs.append(i[0])
#     ys.append(i[1])
#     plt.scatter(xs, ys, s=30, c=colors, alpha=0.5)
# plt.show()

plt.figure(1)

print clusters[0]
for i in range(k):
    colors = np.random.rand(3)
    for j in clusters[i]:
        plt.scatter(j[0], j[1], s=30, c=colors)

print mu
for i in mu:
    colors = [0.5,  0.5,  0.5]
    plt.scatter(i[0], i[1], s=150, c=colors, alpha=0.5)

plt.axis([-1, 1, -1, 1])
plt.show()