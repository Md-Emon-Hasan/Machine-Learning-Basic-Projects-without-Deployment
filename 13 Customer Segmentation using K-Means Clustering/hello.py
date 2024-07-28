import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:,[3,4]].values

# finding wcss value for different number of cluster
wcss = []

for i in range(1,11):
    Kmeans = KMeans(n_clusters=i,init='k-means++')
    Kmeans.fit(x)
    
    wcss.append(Kmeans.inertia_)
    
# training the kmeans clustering model
Kmeans = KMeans(n_clusters=5,init='k-means++')

# return a label for each data point bvased on their cluster
y = Kmeans.fit_predict(x)

# plotring all the cluster and their controls
plt.figure(figsize=(8,8))
plt.scatter(x[y==0,0],x[y==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(x[y==1,0],x[y==1,1],s=50,c='red',label='Cluster 2')
plt.scatter(x[y==2,0],x[y==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(x[y==3,0],x[y==3,1],s=50,c='violet',label='Cluster 4')
plt.scatter(x[y==4,0],x[y==4,1],s=50,c='blue',label='Cluster 5')

# plot the cectroids
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],s=100,c='black',label='Centroids')

plt.title('Cluster Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()