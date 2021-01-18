import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np

boston = sklearn.datasets.load_boston() 
bdata = boston.data

print("start ml")


from sklearn.cluster import KMeans

'''
km = KMeans(10) # 3種類のグループに分ける

data2_clst = km.fit_transform(bdata)
color = ["red", "blue", "green", "yellow", "black","pink","orange","brown","gold","silver"]
for i in range(bdata.shape[0]):
    plt.scatter(data2_clst[i,0], data2_clst[i,1], c=color[int(km.labels_[i])])
plt.show()
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(boston.data)
print(X2D.shape)
print(X2D.shape[0])

km = KMeans(5) # 3種類のグループに分ける
data2_clst = km.fit_transform(X2D)
color = ["red", "blue", "green", "yellow", "black"]
for i in range(X2D.shape[0]):
    plt.scatter(data2_clst[i,0], data2_clst[i,1], c=color[int(km.labels_[i])])
plt.show()