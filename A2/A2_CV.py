import sys
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt

def ReadData(tsv_file):
  data = pd.read_csv(tsv_file, sep="\t", header=None)
  return data

def Normalize(data):
  columns = data.shape[1]
  for i in range(columns-1):
    if (data[i].max() > 1 or data[i].min() < 0):
      tmp_col = data.iloc[:, i]
      data[i] = (tmp_col - tmp_col.min()) / (tmp_col.max()-tmp_col.min())
  return data


def GSCV(data):
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1:]
  train_data, validation_data, train_labels, validation_labels = train_test_split(X, y, test_size=0.1)
  # print(train_data, train_labels, validation_data, validation_labels)
  knn = KNeighborsClassifier()
  k_range = list(range(1, 10))
  weight_options = ['uniform','distance']
  param_gridknn = dict(n_neighbors=k_range, weights=weight_options)
  grid_knn = GridSearchCV(knn, param_gridknn, scoring='accuracy', verbose=1)
  grid_knn.fit(train_data, train_labels.values.ravel())
  res = pd.concat([pd.DataFrame(grid_knn.cv_results_["params"]),pd.DataFrame(grid_knn.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
  print(res)
  print('the best score is: {}'.format(grid_knn.best_score_))
  print('the best param is: {}'.format(grid_knn.best_params_))
  return grid_knn.best_params_

def KFoldCV(data, params, k):
  kf = KFold(n_splits=k, shuffle=True)
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1:]
  for train_index, test_index in kf.split(X):
    train_X, train_y = X.loc[train_index], y.loc[train_index]
    test_X, test_y = X.loc[test_index], y.loc[test_index]
    knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'])
    knn.fit(train_X, train_y.values.ravel())
    score = knn.score(test_X, test_y.values.ravel())
    print('accuracy:{}'.format(score))
    metrics.plot_roc_curve(knn, test_X, test_y.values.ravel())
    metrics.plot_precision_recall_curve(knn, test_X, test_y.values.ravel())
  plt.show()



if __name__ == "__main__":
  if (len(sys.argv) < 2):
    print("Usage: python {} <dataset>".format(sys.argv[0]))
    exit(1)
  data = ReadData(sys.argv[1])
  data = Normalize(data)
  params = GSCV(data)
  KFoldCV(data, params, 5)
