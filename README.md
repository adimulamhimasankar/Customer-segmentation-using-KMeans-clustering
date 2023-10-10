# Customer-segmentation-using-KMeans-clustering

Python Code:

Importing the Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

Data collection and Analysis

#loading the data from csv file to pandas DataFrame


customer_data = pd.read_csv('/Mall_Customers.csv')

#First  5 rows in the dataframe
customer_data.head()

#Finding the number of rows and columns
Customer_data.shape


#getting some information about the data
customer_data.info()


#checking for missing values
customer_data.isnull().sum()

Choosing Annual Income and Spending Score Columns

x = customer_data.iloc[:,[3,4]].values
print(x)

Choosing the number of clusters
Wcss → within clusters sum of squares
Finding wcss value for difference number of cluster

wcss = []


for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(x)


  wcss.append(kmeans.inertia_)
 

Plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow point graph')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()
Optimum Number of clusters = 5
Training the k-means clustering model

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

Return a label for each data point based on their cluster

y = kmeans.fit_predict(x)
print(y)

Visualizing all the cluster
Plotting all the cluster and their centroids
plt.figure(figsize=(8,8))
plt.scatter(x[y==0,0], x[y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(x[y==1,0], x[y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(x[y==2,0], x[y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(x[y==3,0], x[y==3,1], s=50, c='black', label='Cluster 4')
plt.scatter(x[y==4,0], x[y==4,1], s=50, c='brown', label='Cluster 5')




plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='centroids')


plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
