import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train_data_set_x = None
train_data_set_y = None
test_data_set_x = None
test_data_set_y = None

def ReadData(tsv_file):
  data = pd.read_csv(tsv_file, sep="\t", header=0)
  return data

def GSCV():
  clif = Ridge()
  alpha_list = np.logspace(-3, 2, 10)
  norm_list = (True, False)
  lasso_model = GridSearchCV(clif, param_grid={'alpha': alpha_list, 'normalize': norm_list})
  lasso_model.fit(train_data_set_x, train_data_set_y)
  print('best params: ', lasso_model.best_params_)
  y_hat = lasso_model.predict(test_data_set_x)
  mse = np.average((y_hat - np.array(test_data_set_y))**2)
  rmse = np.sqrt(mse)
  print(mse, rmse)

  t = np.arange(len(test_data_set_x))
  plt.plot(t, test_data_set_y, 'r-', linewidth=2, label='Test')
  plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
  plt.legend(loc='upper right')
  plt.grid()
  plt.show()

if __name__ == "__main__":
  if (len(sys.argv) < 2):
    print("Usage: python {} <dataset>".format(sys.argv[0]))
    exit(1)
  data = ReadData(sys.argv[1])
  train_data_set_x, test_data_set_x, train_data_set_y, test_data_set_y = \
    train_test_split(data.iloc[:, :-1], data.iloc[:, -1:], test_size=0.2)
  # clf = Ridge(alpha=1.0)
  # clf.fit(train_data_set_x, train_data_set_y)
  # score = clf.score(test_data_set_x, test_data_set_y.values.ravel())
  # print(score)
  GSCV()