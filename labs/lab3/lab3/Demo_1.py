
#Loading the 'breast cancer' dataset

import numpy as np
import sklearn
#print(sklearn.__version__)

from sklearn import datasets

cancer = datasets.load_breast_cancer()

cancer.keys()

n_features = len(cancer.feature_names)
#print("There are %d features in this dataset" % n_features)
#print("The features are:", cancer.feature_names)

#print(cancer.data.shape)
#print(cancer.data)

#print(cancer.target.shape)
#print(cancer.target)
#print(cancer.target_names)

#Visualising the data

import seaborn as sns
import matplotlib.pyplot as plt

sns.jointplot(cancer.data[:, 0], cancer.data[:, 1])
plt.xlabel(cancer.feature_names[0])

plt.ylabel(cancer.feature_names[1])
#plt.show()

# Insert your own code visualization/analysis here.


# Try to come up with a method that you can use to determine whether your data requires any sort of standarisation.

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf.fit(cancer.data[:-1], cancer.target[:-1])
clf.predict(cancer.data[:-1])
print("Done")

