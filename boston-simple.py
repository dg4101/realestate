import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np

boston = sklearn.datasets.load_boston()

print("start ml")
#dir(boston)
#print(boston.DESCR)

print(boston.data.shape)
 ## (506, 13)
print(boston.feature_names)
 ## ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 ##  'B' 'LSTAT']

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(boston.data)
X2D.shape

print('PCAed Shape' + str(X2D.shape))
print(X2D[:5])


from sklearn.cluster import KMeans
#%matplotlib inline
'''
plt.plot(X2D[:, 0], X2D[:, 1], 'bo')
plt.show()

km = KMeans(5) # 3種類のグループに分ける

data2_clst = km.fit_transform(X2D)
color = ["red", "blue", "green", "yellow", "black"]
for i in range(X2D.shape[0]):
    plt.scatter(X2D[i,0], X2D[i,1], c=color[int(km.labels_[i])])
plt.show(block=False)
'''
#########
from sklearn import datasets
boston1 = datasets.load_boston()

from pandas import DataFrame #pandasモジュール
boston_df = DataFrame(boston1.data) #型をDataFrame型に変換
boston_df.columns = boston1.feature_names #列名の設定
print(boston_df[:5]) #最初の5行だけprintする
boston_df["price"] = boston1.target #住宅価格の追加
print(boston_df[:5]) #最初の5行だけprintする

rooms_train = DataFrame(boston_df["RM"]) #部屋数のデータを抜き出す
print('rooms_train shape' + str(rooms_train.shape))
Y_train = boston1.target #テスト用の教師データ（住宅価格）
from sklearn import linear_model #線形回帰モジュールのインポート
model = linear_model.LinearRegression() #回帰モデル
model.fit(rooms_train, Y_train) #データを入れて訓練

#テストデータの作成
print("Print train min and max")
print(rooms_train.min())
print(rooms_train.max())
print(np.arange(rooms_train.min().values[0], rooms_train.max().values[0], 0.1))

rooms_test=DataFrame(np.arange(rooms_train.min().values[0], rooms_train.max().values[0], 0.1))
print("Print price test")
print(model.predict(rooms_test))
prices_test=model.predict(rooms_test)

plt.scatter(np.array(rooms_train), Y_train, alpha=0.5) #訓練データの散布図
plt.plot(rooms_test.values, prices_test, c="r") #回帰直線
plt.title("Boston House Prices dataset")
plt.xlabel("rooms")
plt.ylabel("prices")
plt.show()
