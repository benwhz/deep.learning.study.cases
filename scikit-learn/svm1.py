from sklearn import svm
import numpy as np

X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 1, 0, 0]
clf = svm.LinearSVC()
clf.fit(X, y)

print(clf.predict([[2., -2.]]))

a = np.array([2, 2])
b = np.array([2, 1])

print(a@b)