from sklearn import datasets
boston = datasets.load_boston()

print(dir(boston))

print(boston.DESCR)

print(boston.feature_names)

print(boston.target)

from pandas import DataFrame #pandasモジュール
boston_df = DataFrame(boston.data) #型をDataFrame型に変換
boston_df.columns = boston.feature_names #列名の設定
boston_df["price"] = boston.target #住宅価格の追加
print(boston_df[:5]) #最初の5行だけprintする

####################################
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
model = KMeans(10) # x種類のグループに分ける
model.fit(boston_df)

print(model.labels_)

markers = ["+", "*", "o","+", "*", "o","+", "*", "o","+"]
for i in range(10):
    p = boston.data[model.labels_ == i, :]
    plt.scatter(p[:, 0], p[:, 1], marker = markers[i], color = 'r')

plt.show()
