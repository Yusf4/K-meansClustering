# This is a sample Python script.

import numpy  as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
from  sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

data ,true_labels=make_blobs(n_samples=100,centers=3,cluster_std=1.0,random_state=42)
# initialize  the Kmeans Model
kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(data)
labels=kmeans.labels_
centroids=kmeans.cluster_centers_
print("Cluster labels:",labels)
print("Centroids:",centroids)

plt.scatter(data[:,0],data[:,1],c=labels,cmap='viridis',label='data Points')
plt.scatter(centroids[:,0],centroids[:, 1],s=300,c='red',marker='X',label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Script executed successfully!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
