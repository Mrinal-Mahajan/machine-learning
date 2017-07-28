import numpy as np
from scipy import ndimage
from time import time
from sklearn import datasets, manifold
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GMM
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = datasets.load_iris()
X,y = iris.data[:,:2], iris.target

num_clusters = 8
model = KMeans(n_clusters=num_clusters)
model.fit(X)

labels = model.labels_
cluster_centers = model.cluster_centers_
print cluster_centers

plt.scatter(np.array(X[:,0]), np.array(X[:,1]),c=labels.astype(np.float))
plt.hold(True)
plt.scatter(np.array(cluster_centers[:,0]), np.array(cluster_centers[:,1]), c = np.arange(num_clusters), marker ='^', s = 150)
plt.show()
plt.scatter(np.array(X[:,0]), np.array(X[:,1]),c=np.choose(y,[0,2,1]).astype(np.float))
plt.show()
