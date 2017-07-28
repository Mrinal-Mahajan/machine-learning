from __future__ import division, print_function
import numpy as np
from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def evaluate_on_test_data(model=None):
	predictions = model.predict(X_test)
	correct_classifications = 0
	for i in range(len(y_test)):
		if predictions[i] == y_test[i]:
			correct_classifications += 1
	accuracy = 100*correct_classifications/len(y_test) #Accuracy as a percentage
	return accuracy

kernels = ['linear','poly','rbf']
accuracies = []
for kernel in kernels:
	model = svm.SVC(kernel=kernel)
	model.fit(X_train, y_train)
	acc = evaluate_on_test_data(model)
	accuracies.append(acc)
	print("{} % accuracy obtained with kernel = {}".format(acc, kernel))

#Train SVMs with different kernels
svc = svm.SVC(kernel='linear').fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7).fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)
#Create a mesh to plot in
h = .02 # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
#Define title for the plots
titles = ['SVC with linear kernel','SVC with RBF kernel','SVC with polynomial (degree 3) kernel']
for i,clf in enumerate((svc, rbf_svc, poly_svc)):
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, m_max]x[y_min, y_max].
	plt.figure(i)
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
	# plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)
	# Plot also the training points
	plt.scatter(np.array(X[:, 0]), np.array(X[:, 1]), c=y, cmap=plt.cm.ocean)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title(titles[i])
	plt.show()

for clf in (svc, rbf_svc, poly_svc):
	print("The support vectors for are:\n", clf.support_vectors_)
	print("The no. of support vectors = {}".format(len(clf.support_vectors_)))

