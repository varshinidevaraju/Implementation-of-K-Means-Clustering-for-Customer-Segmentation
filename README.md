# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and matplotlib.pyplot
2. Read the dataset and transform
3. Import KMeans and fit the data in the model
4. Plot the Cluster graph


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: D.VARSHINI
RegisterNumber:  212223230234
*/

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('Exp_9_Mall_Customers.csv')

data.head()
data.info()
data.isnull().sum()

from sklearn.cluster import KMeans
wcss =[] #Within-cluster Sum pof Square.
#It is the sum of Squared distance between each point & the centroid in the cluster

for i in range(1,11):
    kmeans=KMeans(n_clusters = i, init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow method")

km = KMeans(n_clusters =5)
km.fit(data.iloc[:,3:])

KMeans(n_clusters=5)

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="black",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="orange",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="red",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
## Elbow Method:
![image](https://github.com/user-attachments/assets/0770e4bb-fc9d-45ff-a2b6-445124ef8193)


## Y-Predict:
![image](https://github.com/user-attachments/assets/2885f71b-c125-4d81-b6be-a8b50a155009)


## Customer Segments:

![image](https://github.com/user-attachments/assets/9b18ec2e-f32f-4560-a7c8-327f7fcfd2ab)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
