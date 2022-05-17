#%%
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# %%
######################################################
######################################################
filename = "../data/sample1.png"
# filename = "../data/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png"
import numpy as np
from PIL import Image
imageBN = Image.open(filename).convert('L')
data = np.asarray(imageBN)
# print(type(data))
# print(data.shape)
rows, cols = data.shape

import matplotlib.pyplot as plt
plt.imshow(data, cmap='gray', vmin=0, vmax=255)

#%%
data_ = data.copy()

threshold = 30  ## to improve here ***
data_[ data_ < threshold ] = 0
data_[ data_ >= threshold ] = 255

# threshold = 90  ## to improve here ***
# data_[ data_ < threshold ] = 0
# data_[ data_ >= threshold ] = 255

plt.imshow(data_, cmap='gray', vmin=0, vmax=255)

#%%
X_ = []
# x = []
# y = []
for r in range(rows):
    for c in range(cols):
        if data_[r, c] >= threshold:
            X_.append([r, c])
            # x.append(r)
            # y.append(c)
#
X_ = np.asarray(X_)

import matplotlib.pyplot as plt
plt.plot(X_[:, 1], X_[:, 0],  "b.")
# plt.plot(y, x, col + ".")
# plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.xlim(0, cols)
plt.ylim(0, rows)
ax = plt.gca()
ax.invert_yaxis()
plt.show()


#%%

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X_, quantile=0.1, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X_)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


#%%
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    # print(my_members)
    plt.plot(X_[my_members, 1], X_[my_members, 0], col + ".")
    # plt.plot(
    #     cluster_center[1],
    #     cluster_center[0],
    #     "o",
    #     markerfacecolor=col,
    #     markeredgecolor="k",
    #     markersize=14,
    # )

#
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.xlim(0, cols)
plt.ylim(0, rows)
ax = plt.gca()
ax.invert_yaxis()
plt.show()


# %%
import matplotlib.pyplot as plt
from itertools import cycle

data_background = np.zeros((rows + 1, cols))

plt.figure(1)
plt.clf()

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    # print(my_members)
    plt.plot(X_[my_members, 1], X_[my_members, 0], col + ".")
    # plt.plot(
    #     cluster_center[1],
    #     cluster_center[0],
    #     "o",
    #     markerfacecolor=col,
    #     markeredgecolor="k",
    #     markersize=14,
    # )
    #
    xlist = X_[my_members, 1]
    ylist = X_[my_members, 0]
    data_new = data_background.copy()
    for (i, j) in zip(xlist, ylist):
        data_new[j, i] = 255
    #
    img2 = Image.fromarray(data_new)
    if img2.mode != 'RGB':
        img2 = img2.convert('RGB')
    #
    img2.save( "data/" + str(k + 1) + ".jpeg" )

#
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.xlim(0, cols)
plt.ylim(0, rows)
ax = plt.gca()
ax.invert_yaxis()
plt.show()


# %%

