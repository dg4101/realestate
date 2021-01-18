from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy import stats

def bs_load_data(trim):
      # datasetのインスタンスを宣言
      boston = load_boston()
      #boston = load_wine()
      #　説明変数と目的変数を生成
      X = boston.data
      X = stats.trimboth(X, trim)
      y = boston.target
      y = stats.trimboth(y, trim)
      # Index name
      fn = boston.feature_names
      # データを分割
      X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100,test_size=0.20)


      print('データセットのレコード数: ', len(X), '\n',
            'トレーニングデータのレコード数: ', len(X_train), '\n',
            'テストデータのレコード数: ', len(X_test))

      print("X:" + str(X.shape))
      print("X:" + str(X[:2]))
      print("X_train:" + str(X_train.shape))
      print("X_test:" + str(X_test.shape))
      print("y:" + str(y.shape))
      print("y:" + str(y[:2]))
      print("y_train:" + str(y_train.shape))
      print("y_test:" + str(y_test.shape))

      return X_train, X_test, y_train, y_test, fn