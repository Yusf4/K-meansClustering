# This is a sample Python script.

import numpy  as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
from  sklearn.cluster import KMeans


# create the dataset example

data=np.array([
     [1, 2], [2, 1], [3, 4], [5, 7],
     [3, 2], [8, 9], [9, 8], [7, 6]
 ])
# initialize  the Kmeans Model
kmeans=KMeans(n_clusters=2,random_state=42)
kmeans.fit(data)
labels=kmeans.labels_
centroids=kmeans.cluster_centers_
print("Cluster labels:",labels)
print("Centroids:",centroids)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Script executed successfully!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
