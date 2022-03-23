import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


# Importing the dataset
data = pd.read_csv('EM/xclara.csv')
print("Input Data and Shape")
print(data.shape)
#print(data.head())


# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
print(f1)
X = np.array(list(zip(f1, f2)))
'''print(f1)
print('----------------')
print(X)'''
print('Graph for whole dataset')
plt.scatter(f1, f2, c='black', s=7)
plt.title('whole dataset')
plt.show()
##########################################


kmeans = KMeans(3)
labels = kmeans.fit(X).predict(X)
centroids = kmeans.cluster_centers_
#print(centroids)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=7)
print('Graph using Kmeans Algorithm')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
'''print(centroids[:, 0])
print(centroids[:, 1])'''
plt.title('kmeans')
plt.show()



#gmm demo
gmm = GaussianMixture(3).fit(X)
labels = gmm.predict(X)
#for ploting
#probs = gmm.predict_proba(X)
#size = 10 * probs.max(1) ** 3
print('Graph using EM Algorithm')
#print(probs[:300].round(4))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=7)
plt.title('em')
plt.show()