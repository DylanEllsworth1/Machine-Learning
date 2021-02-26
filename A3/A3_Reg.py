import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

def ReadData(tsv_file):
  data = pd.read_csv(tsv_file, sep="\t", header=0)
  return data

def KFoldCV(data, k):
  kf = KFold(n_splits=k, shuffle=True)
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1:]
  for train_index, test_index in kf.split(X):
    train_X, train_y = X.loc[train_index], y.loc[train_index]
    test_X, test_y = X.loc[test_index], y.loc[test_index]
    reg = LinearRegression().fit(train_X, train_y)
    score = reg.score(test_X, test_y)
    result = reg.predict(test_X)
    R2 = r2_score(result, test_y)
    MSE = mean_squared_error(result, test_y)
    RSS = MSE*test_X.shape[0]
    print('coef:{}, intercept:{}\nscore:{}, R2:{}, RSS:{}\n'.format(reg.coef_, reg.intercept_, \
      score, R2, RSS))

if __name__ == "__main__":
  if (len(sys.argv) < 2):
    print("Usage: python {} <dataset>".format(sys.argv[0]))
    exit(1)
  data = ReadData(sys.argv[1])
  KFoldCV(data, 10)
