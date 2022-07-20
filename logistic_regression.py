from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
x = iris["data"][:, 3:]  # 3rd column
y = (iris["target"] == 2).astype(np.int)
print(x)  # here we take only one feature as input
print(y)  # it is the output based on input
# print(list(iris.keys()))
# print(iris['data'])
# print(iris['target'])
# print(iris['DESCR'])

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x, y)
ex = clf.predict([[1.8]])
print(ex)

# using matplot lib

X_new = np.linspace(0, 3, 1000).reshape(1000, 1)
# print(X_new)
A_prob = clf.predict_proba(X_new)
plt.plot(X_new, A_prob[:, 1])
plt.show()
