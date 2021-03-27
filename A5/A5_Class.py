import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def ReadData(tsv_file):
  data = pd.read_csv(tsv_file, sep="\t", header=0)
  return data

def KFoldCV(model, data_X, data_y, k):
  kf = KFold(n_splits=k, shuffle=True)
  for train_index, test_index in kf.split(data_X):
    train_X = data_X[train_index]
    train_y = data_y[train_index]
    test_X, test_y = data_X[test_index], data_y[test_index]
    model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    print('accuracy:{}'.format(score))

def GraphicalPerformance(model):
    metrics.plot_roc_curve(model, X_test, y_test.ravel())
    metrics.plot_precision_recall_curve(clf, X_test, y_test.ravel())
    plt.show()


if __name__ == "__main__":
  if (len(sys.argv) < 2):
    print("Usage: python {} <dataset>".format(sys.argv[0]))
    exit(1)
  data = ReadData(sys.argv[1])
  print(data.shape)
  X_train, X_test, y_train, y_test = \
    train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1:].values, test_size=0.1)
  # SVM model
  # choose best kernel function
  param_grid = {'kernel': ['rbf', 'linear', 'poly']}
  clf = SVC()
  GS = GridSearchCV(clf, param_grid, cv=10)
  GS.fit(X_train, y_train.ravel())
  print(GS.best_params_)
  print(GS.best_score_)
  # choose C and gamma
  param_grid = {
    'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
  }
  clf = SVC(kernel=GS.best_params_['kernel'])
  GS = GridSearchCV(clf, param_grid, cv=10)
  GS.fit(X_train, y_train.ravel())
  print(GS.best_params_)
  print(GS.best_score_)
  # K-fold CV
  clf = SVC(kernel='rbf', C=11, gamma=0.00001)
  print('SVM: ')
  KFoldCV(clf, X_train, y_train.ravel(), 10)
  # show performance
  metrics.plot_roc_curve(clf, X_test, y_test.ravel())
  metrics.plot_precision_recall_curve(clf, X_test, y_test.ravel())


  # Random Forest
  # search for best n_estimators
  scores = []
  for i in range(0, 200, 10):
      rfc = RandomForestClassifier(n_estimators = i+1, n_jobs=-1, random_state=90)
      score = cross_val_score(rfc, X_train, y_train.ravel(), cv=10).mean()
      scores.append(score)
  print(max(scores), scores.index(max(scores))*10+1)
  # plt.figure(figsize=[20, 5])
  # plt.plot(range(1, 201, 10), scores)
  # plt.show()

  # max_depth
  param_grid = {'max_depth':np.arange(1, 20, 1)}
  rfc = RandomForestClassifier(n_estimators = 161, random_state=90)
  GS = GridSearchCV(rfc, param_grid, cv=10)
  GS.fit(X_train, y_train.ravel())
  print(GS.best_params_)
  print(GS.best_score_)

  # min_samples_leaf
  param_grid={'min_samples_leaf':np.arange(1, 1+10, 1)}
  rfc = RandomForestClassifier(n_estimators = 161, max_depth=17, random_state=90)
  GS = GridSearchCV(rfc, param_grid, cv=10)
  GS.fit(X_train, y_train.ravel())
  print(GS.best_params_)
  print(GS.best_score_)

  # min_samples_split
  param_grid={'min_samples_split':np.arange(2, 2+20, 1)}
  rfc = RandomForestClassifier(n_estimators = 161, max_depth=17, min_samples_leaf=1, random_state=90)
  GS = GridSearchCV(rfc,param_grid,cv=10)
  GS.fit(X_train, y_train.ravel())
  print(GS.best_params_)
  print(GS.best_score_)

  # criterion
  param_grid = {'criterion':['gini', 'entropy']}
  rfc = RandomForestClassifier(n_estimators = 161, max_depth=17, min_samples_leaf=1, random_state=90)
  GS = GridSearchCV(rfc,param_grid,cv=10)
  GS.fit(X_train, y_train.ravel())
  print(GS.best_params_)
  print(GS.best_score_)

  # K-vold CV for RF
  rfc = RandomForestClassifier(n_estimators = 161, max_depth=17, min_samples_leaf=1, criterion='gini', min_samples_split=2, random_state=90)
  KFoldCV(clf, X_train, y_train.ravel(), 10)

  # performance of RF
  metrics.plot_roc_curve(rfc, X_test, y_test.ravel())
  metrics.plot_precision_recall_curve(clf, X_test, y_test.ravel())

  # predict A5_test
  # final SVM model
  # clf = SVC(probability=True, kernel='rbf', C=11, gamma=0.00001)
  # clf.fit(data.iloc[:, :-1].values, data.iloc[:, -1:].values.ravel())
  # A5_X = ReadData("A5_test_dataset.tsv")
  # A5_p = clf.predict_proba(A5_X)
  # with open('A5_predictions_group47.txt', 'w') as f:
  #     for i in range(A5_p.shape[0]):
  #         f.write(str(A5_p[i][1])+"\n")