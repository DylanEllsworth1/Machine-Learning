import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


train_data_set_x = None
train_data_set_y = None
test_data_set_x = None
test_data_set_y = None
data_X = None
data_y = None

def ReadData(tsv_file):
  data = pd.read_csv(tsv_file, sep="\t", header=0)
  return data

def GSCV():
  clif = Ridge()
  alpha_list = np.logspace(-3, 2, 10)
  norm_list = (True, False)
  ridge_model = GridSearchCV(clif, param_grid={'alpha': alpha_list, 'normalize': norm_list})
  ridge_model.fit(train_data_set_x, train_data_set_y)
  print('best params: ', ridge_model.best_params_)
  return ridge_model.best_params_

def KFoldCV(model, data_X, data_y, k):
  kf = KFold(n_splits=k, shuffle=True)
  accuracy_tot = 0
  R2_tot = 0
  RSS_tot = 0
  for train_index, test_index in kf.split(train_data_set_x):
    train_X = train_data_set_x[train_index]
    train_y = train_data_set_y[train_index]
    test_X, test_y = train_data_set_x[test_index], train_data_set_y[test_index]
    model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    y_hat = model.predict(test_X)
    R2 = r2_score(y_hat, test_y)
    R2_tot = R2_tot + R2
    MSE = mean_squared_error(y_hat, test_y)
    RSS = MSE*test_X.shape[0]
    RSS_tot = RSS_tot + RSS
    accuracy_tot = accuracy_tot + score
    print('accuracy:{}, R2:{}, RSS:{}'.format(score, R2, RSS))
  print('average-> accuracy: {}, R2:{}, RSS:{}'.format(accuracy_tot/k, R2_tot/k, RSS_tot/k))


if __name__ == "__main__":
  if (len(sys.argv) < 2):
    print("Usage: python {} <dataset>".format(sys.argv[0]))
    exit(1)
  data = ReadData(sys.argv[1])
  train_data_set_x, test_data_set_x, train_data_set_y, test_data_set_y = \
    train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1:].values, test_size=0.1)
  data_X = data.iloc[:, :-1].values
  data_y = data.iloc[:, -1:].values
  poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
  data_X = poly.fit_transform(data_X)
  train_data_set_x = poly.fit_transform(train_data_set_x)
  test_data_set_x = poly.fit_transform(test_data_set_x)
  #Grid search cv
  params = GSCV()
  model = Ridge(alpha=params['alpha'], normalize=params['normalize'])
  # 10 fold CV
  KFoldCV(model, data_X, data_y, 10)
  # test
  model.fit(train_data_set_x, train_data_set_y)
  y_hat = model.predict(test_data_set_x)
  mse = mean_squared_error(y_hat, test_data_set_y)
  RSS = mse * len(test_data_set_y)
  R2 = r2_score(y_hat, test_data_set_y)
  print(R2, RSS)
  # t = np.arange(len(test_data_set_x))
  # plt.plot(t, test_data_set_y, 'r-', linewidth=2, label='Test')
  # plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
  # plt.legend(loc='upper right')
  # plt.grid()
  # plt.show()

  # train final model
  model = Ridge(alpha=params['alpha'], normalize=params['normalize'])
  model.fit(data_X, data_y)
  print(model.coef_)
  # data_X = ReadData("./A4_TestData.tsv")
  # data_X = poly.fit_transform(data_X)
  # print(data_X.shape)
  # y_hat = model.predict(data_X)
  # with open('A4_predictions_group47.txt', 'w') as f:
  #   f.write(str(y_hat))


