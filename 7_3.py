import numpy as np
from scipy import ndimage
from time import time
from sklearn import datasets, manifold
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GMM
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib as mpl

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
np.random.seed(0)
def nudge_images(X, y):
	# Having a larger dataset shows more clearly the behavior of the
	# methods, but we multiply the size of the dataset only by 2, as the
	# cost of the hierarchical clustering methods are strongly
	# super-linear in n_samples
	shift = lambda x: ndimage.shift(x.reshape((8, 8)),.3 * np.random.normal(size=2),mode='constant',).ravel()
	X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
	Y = np.concatenate([y, y], axis=0)
	return X, Y
X, y = nudge_images(X, y)

def plot_clustering(X_red, X, labels, title=None):
	x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
	X_red = (X_red - x_min) / (x_max - x_min)
	plt.figure(figsize=(2*6, 2*4))
	for i in range(X_red.shape[0]):
		plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),color=plt.cm.spectral(labels[i] / 10.),fontdict={'weight': 'bold', 'size': 9})
	plt.xticks([]) 
	plt.yticks([])
	if title is not None:
		plt.title(title, size=17)
	plt.axis('off')
	plt.tight_layout()

print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

for linkage in ('ward', 'average', 'complete'):
	clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
	t0 = time()
	clustering.fit(X_red)
	print("%s : %.2fs" % (linkage, time() - t0))
	plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)
plt.show()