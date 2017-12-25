# Train and fine-tune a decision tree for the moons datasets.
# Grow a random forest.
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import mode


# Generate a moons dataset.
data = datasets.make_moons(n_samples=10000, noise=0.4)

# Split the dataset.
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.7, random_state=42)

# Use grid search with cross-validation to find good
# hyperparameter values for DecisionTreeClassifier.
param_grid = {'max_leaf_nodes': range(2, 8)}
clf = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)
clf.fit(X, y)
print(clf.best_params_)
best_max_leaf_nodes = clf.best_params_['max_leaf_nodes']

# Train with a decision tree model using the found hyperparameter above.
clf = DecisionTreeClassifier(max_leaf_nodes=best_max_leaf_nodes)
clf.fit(X_train, y_train)
score = accuracy_score(clf.predict(X_test), y_test)
print(score)

#
# Grow a random forest with DecisionTreeClassifier.
#
len_data = len(data[0])
scores = []
clfs = []

for i in range(1000):
    # Generate 1000 subsets of the training set,
    # each containing 100 instances selected randomly.
    random_index = np.random.choice(range(len_data), size=100, replace=False)
    X_train, y_train = data[0][random_index], data[1][random_index]
    # Train one decision tree on each subset using the best
    # hyperparameter values found above.
    clf = DecisionTreeClassifier(max_leaf_nodes=4).fit(X_train, y_train)
    score = accuracy_score(clf.predict(X_test), y_test)
    scores.append(score)
    clfs.append(clf)
# Averaged scores from 1000 decision trees.
score = np.mean(scores)

# For each test set, generate the predictions of the
# 1000 decision trees from above.
y_predicts = []
for i, clf in enumerate(clfs):
    y_predict = clf.predict(X_test)
    y_predicts.append(y_predict)
y_predict = np.vstack(y_predicts)
# Keep the most frequent predictions. (majority-vote predictions)
voted_y_predict = mode(y_predict)[0][0]
score = accuracy_score(voted_y_predict, y_test)
print(score)