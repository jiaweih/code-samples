# Use VotingClassifier to combine a few classifiers.
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

data = datasets.make_moons(n_samples=10000, noise=0.4)

# Split the dataset.
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.7, random_state=42)

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Use GridSearchCV to find the best parameters.
params = {'rf__n_estimators': [50, 100],
          'svc__C': [0.4, 0.5, 0.6],
          'lr__C': [0.5, 0.8, 1.0]}
grid = GridSearchCV(estimator=voting_clf, param_grid=params, n_jobs=-1)
grid = grid.fit(X_train, y_train)
print(grid.best_params_)
