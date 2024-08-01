#!/usr/bin/env python
# coding: utf-8

# In[1228]:


# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as mtp 
import matplotlib.cm as cm
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score as sil, calinski_harabasz_score as chs, silhouette_samples
from kneed import KneeLocator





# In[1229]:


df = pd.read_excel('EastWestAirlines.xlsx', sheet_name='data')


# In[1230]:


df


# In[1231]:


df


# In[1232]:


df.rename(columns={'ID#':'ID', 'Award?':'Award'}, inplace=True)


# In[1233]:


df.set_index('ID',inplace=True)
df


# In[1234]:


print("unique_cc1",df.cc1_miles.unique())
print("unique_cc2",df.cc2_miles.unique())
print("unique_cc3",df.cc3_miles.unique())


# In[1235]:


df.isnull().sum()


# In[1236]:


print("unique Award",df.Award.unique())


# In[1237]:


for feature in data.columns:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()


# In[1238]:


df.columns


# In[1239]:


outlier = df.copy() 
fig, axes=plt.subplots(10,1,figsize=(8,9),sharex=False,sharey=False)
sns.boxplot(x='Balance',data=outlier,palette='crest',ax=axes[0])
sns.boxplot(x='Qual_miles',data=outlier,palette='crest',ax=axes[1])
sns.boxplot(x='cc1_miles',data=outlier,palette='crest',ax=axes[2])
sns.boxplot(x='cc2_miles',data=outlier,palette='crest',ax=axes[3])
sns.boxplot(x='cc3_miles',data=outlier,palette='crest',ax=axes[4])
sns.boxplot(x='Bonus_miles',data=outlier,palette='crest',ax=axes[5])
sns.boxplot(x='Bonus_trans',data=outlier,palette='crest',ax=axes[6])
sns.boxplot(x='Flight_miles_12mo',data=outlier,palette='crest',ax=axes[7])
sns.boxplot(x='Flight_trans_12',data=outlier,palette='crest',ax=axes[8])
sns.boxplot(x='Days_since_enroll',data=outlier,palette='crest',ax=axes[9])
plt.tight_layout(pad=0.2)


# In[1240]:


# Box plot for every feature in the same graph

plt.figure(figsize=(12,8))
sns.boxplot(data=df)


# In[1241]:


# we use sqrt() to see more clearly despite the outliers

plt.figure(figsize=(12,8))
sns.boxplot(data=np.sqrt(df))


# In[1242]:


countnotermdeposit = len(df[df.Award == 0])


# In[1243]:


countnotermdeposit 


# In[1244]:


print("customer have not award",(countnotermdeposit / len(df.Award)*100))


# In[1245]:


counthavetermdeposit = len(df[df.Award == 1])
counthavetermdeposit


# In[1246]:


print("customer have a award",(counthavetermdeposit / len(df.Award)*100))


# In[1247]:


sns.countplot(data=df,x='Award')
plt.xticks(fontsize = 12)
plt.title('client has award or not')
plt.show()


# In[1248]:


Balance = df[['Award','Balance']]
sns.barplot(data=Balance,x='Award',y='Balance')
plt.xticks(fontsize = 12)
plt.xlabel('Award')
plt.ylabel('Balance')
plt.title('Balance: no of miles eligible for award travel')
plt.show()


# In[1249]:


corr_matrix =  df.corr()
corr_matrix['Balance'].sort_values(ascending=False)


# In[1250]:


f,ax = plt.subplots(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, linewidths = .5, fmt ='.1f', ax=ax)
plt.show()


# In[1251]:


# Plotting frequent flying bonuses vs. non-flight bonus transactions 
plt.figure(figsize = (10,10))
sorted_data = df[['cc1_miles','Bonus_trans']].sort_values('Bonus_trans', ascending = False)
ax = sns.barplot(x='cc1_miles', y='Bonus_trans', data = sorted_data)
ax.set(xlabel = 'Miles earned with freq. flyer credit card', ylabel= 'Non-flight bonus transactions')
plt.xticks(rotation=90)
plt.show()


# In[1252]:


standard_scaler = StandardScaler()
std_df = standard_scaler.fit_transform(df)
std_df.shape


# In[1253]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

minmax_df = minmax.fit_transform(df)
minmax_df.shape


# In[1256]:


cluster_range = range(1,15)
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters,n_init=10)
    clusters.fit(std_df)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_
    cluster_errors.append(clusters.inertia_)
clusters_df = pd.DataFrame({"Num_Clusters":cluster_range,"Cluster_Errors":cluster_errors})
clusters_df


# In[1257]:


wcss=[]
for i in range(1,9):
    kmeans = KMeans(n_clusters=i,random_state=2)
    kmeans.fit(std_df)
    wcss.append(kmeans.inertia_)
    
# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,9),wcss,color = 'black')
plt.scatter(range(1,9),wcss,color='red')
plt.title('Elbow Graph for Standard Scaler')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[1227]:


kl = KneeLocator(range(1,9),wcss,curve='convex',direction='decreasing')
kl.elbow


# In[1074]:


from sklearn.metrics import silhouette_score


# In[1258]:


from sklearn.metrics import silhouette_score
n_clusters = [2,3,4,5,6,7,8,9,10] # number of clusters
clusters_inertia = [] # inertia of clusters
s_scores = [] # silhouette scores

for n in n_clusters:
    KM_est = KMeans(n_clusters=n, init='k-means++').fit(std_df)
    clusters_inertia.append(KM_est.inertia_)    # data for the elbow method
    silhouette_avg = silhouette_score(std_df, KM_est.labels_)
    s_scores.append(silhouette_avg) # data for the silhouette score method


# In[1259]:


fig, ax = plt.subplots(figsize=(12,5))
ax = sns.lineplot(x= n_clusters, y = s_scores, marker='o', ax=ax)
ax.set_title("Silhouette score method")
ax.set_xlabel("number of clusters")
ax.set_ylabel("Silhouette score")
ax.axvline(2, ls="--", c="red")
plt.grid()
plt.show()


# In[1260]:


# Instantiate a scikit-learn K-Means model. we will check for two diff hyperparameters value effect.
model = KMeans(random_state=10, max_iter=500, init='k-means++')

# Instantiate the KElbowVisualizer with the number of clusters and the metric
visualizer = KElbowVisualizer(model, k=(2,20), metric='silhouette', timings=False)
fig, ax = plt.subplots(figsize=(12,5))
# Fit the data and visualize
print('Elbow Plot for Standard Scaler data')
visualizer.fit(std_df)    
visualizer.poof()
plt.show()


# In[1261]:


# 1. How many number of clusters? n_clusters?

# Since true labels are not known..we will use Silhouette Coefficient (Clustering performance evaluation)
# knee Elbow graph method


# Instantiate a scikit-learn K-Means model. we will check for two diff hyperparameters value effect.
model = KMeans(random_state=10, max_iter=500, init='k-means++')


# In[1262]:


# Instantiate the KElbowVisualizer with the number of clusters and the metric
visualizer = KElbowVisualizer(model, k=(2,20), metric='silhouette', timings=False)


# In[1263]:


# Fit the data and visualize
print('Elbow Plot for Standard Scaler data')
visualizer.fit(std_df)    
visualizer.poof()
plt.show()


# In[1268]:


# With the elbow method, the ideal number of clusters to use was 5.
# We will also use the Silhouette score to determine an optimal number.clust_list = [2,3,4,5,6,7,8,9]

# Silhouette score for standered scaler Applied on data .
for n_clusters in clust_list:
    clusterer1 = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels1 = clusterer1.fit_predict(std_df)
    sil_score1= silhouette_score(std_df, cluster_labels1)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", sil_score1)


# In[1269]:


#higher silhouette_score is : 0.3353447174269873 for n_cluster = 5 


# In[1270]:


model_kmeans = KMeans(n_clusters=5, random_state=0, init='k-means++')
y_predict_kmeans = model_kmeans.fit_predict(std_df)
y_predict_kmeans.shape


# In[1271]:


y_predict_kmeans


# In[1272]:


model_kmeans.labels_


# In[1273]:


# cluster centres associated with each lables

model_kmeans.cluster_centers_


# In[1274]:


model_kmeans.inertia_


# In[1275]:


df = pd.read_excel('EastWestAirlines.xlsx', sheet_name='data')
df.rename({'ID#':'ID', 'Award?':'Award'}, inplace=True, axis=1)
df['Kmeans_label'] = model_kmeans.labels_


# In[1276]:


df.groupby('Kmeans_label').mean()


# In[1277]:


fig, ax = plt.subplots(figsize=(10, 5))
df.groupby(['Kmeans_label']).count()['ID'].plot(kind='bar', ax=ax)
plt.title('Kmeans Clustering Standard Scaler Applied', fontsize='large', fontweight='bold')
ax.set_xlabel('Clusters', fontsize='medium', fontweight='bold')
ax.set_ylabel('ID Counts', fontsize='medium', fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[1278]:


cluster_range = range(1,15)
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters,n_init=10)
    clusters.fit(minmax_df)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_
    cluster_errors.append(clusters.inertia_)
clusters_df = pd.DataFrame({"Num_Clusters":cluster_range,"Cluster_Errors":cluster_errors})
clusters_df


# In[1279]:


wcss=[]
for i in range (1,9):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(minmax_df)
    wcss.append(kmeans.inertia_)
    
# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,9),wcss,color = 'black')
plt.scatter(range(1,9),wcss,color='red')
plt.title('Elbow Graph for MinMaxScaler')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[1280]:


from sklearn.metrics import silhouette_score
n_clusters = [2,3,4,5,6,7,8,9,10] # number of clusters
clusters_inertia = [] # inertia of clusters
s_scores = [] # silhouette scores

for n in n_clusters:
    KM_est = KMeans(n_clusters=n, init='k-means++').fit(minmax_df)
    clusters_inertia.append(KM_est.inertia_)    # data for the elbow method
    silhouette_avg = silhouette_score(minmax_df, KM_est.labels_)
    s_scores.append(silhouette_avg) # data for the silhouette score method

fig, ax = plt.subplots(figsize=(12,5))
ax = sns.lineplot(x = n_clusters,y = s_scores, marker='o', ax=ax)
ax.set_title("Silhouette score method")
ax.set_xlabel("number of clusters")
ax.set_ylabel("Silhouette score")
ax.axvline(2, ls="--", c="red")
plt.grid()
plt.show()


# In[1281]:


# With the elbow method, the ideal number of clusters to use was 2.
# We will also use the Silhouette score to determine an optimal number.

clust_list = [2,3,4,5,6,7,8,9]

#  Silhouette score for MinMaxScaler Applied on data .

for n_clusters in clust_list:
    clusterer1 = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels1 = clusterer1.fit_predict(minmax_df)
    sil_score1= sil(minmax_df, cluster_labels1)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", sil_score1)


# In[1089]:


# we have found good number of cluster = 2
# model building using cluster numbers = 2

model_kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++')
y_predict_kmeans = model_kmeans.fit_predict(minmax_df)
y_predict_kmeans.shape


# In[1090]:


y_predict_kmeans


# In[1283]:


# cluster centres associated with each lablesmodel_kmeans.labels_model_kmeans.cluster_centers_

model_kmeans.cluster_centers_


# In[1284]:


# within-cluster sum of squared

# The lower values of inertia are better and zero is optimal.
# Inertia is the sum of squared error for each cluster. 
# Therefore the smaller the inertia the denser the cluster(closer together all the points are)

model_kmeans.inertia_


# In[1285]:


#Assign clusters to the data set
data = pd.read_excel('EastWestAirlines.xlsx', sheet_name='data')
data.rename({'ID#':'ID', 'Award?':'Award'}, inplace=True, axis=1)
data['Kmeans_label'] = model_kmeans.labels_


# In[1286]:


# Plotting barplot using groupby method to get visualize how many row no. in each cluster

fig, ax = plt.subplots(figsize=(10, 6))
data.groupby(['Kmeans_label']).count()['ID'].plot(kind='bar')
plt.ylabel('ID Counts')
plt.title('Kmeans Clustering minmax Applied',fontsize='large',fontweight='bold')
ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')
ax.set_ylabel('ID counts', fontsize='large', fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[1287]:


# Group data by Clusters (K=2)
data.groupby('Kmeans_label').agg(['mean'])


# In[1288]:


#
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

n_clusters = [2, 3, 4, 5, 6, 7, 8]  # always start number from 2.
linkages = 'ward'



for n in n_clusters:
        hie_cluster1 = AgglomerativeClustering(n_clusters=n,linkage=linkages) # bydefault it takes linkage 'ward'
        hie_labels1 = hie_cluster1.fit_predict(minmax_df)
        silhouette_score1 = silhouette_score(minmax_df, hie_labels1)
        print("For n_clusters =", n,"The average silhouette_score with linkage-",linkages, ':',silhouette_score1)
    


# In[1289]:


#
n_clusters = [2, 3, 4, 5, 6, 7, 8]  # always start number from 2.
linkages = 'ward'



for n in n_clusters:
        hie_cluster1 = AgglomerativeClustering(n_clusters=n,linkage=linkages) # bydefault it takes linkage 'ward'
        hie_labels1 = hie_cluster1.fit_predict(std_df)
        silhouette_score1 = silhouette_score(std_df, hie_labels1)
        print("For n_clusters =", n,"The average silhouette_score with linkage-",linkages, ':',silhouette_score1)


# In[1290]:


agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
y_pred_hie = agg_clustering.fit_predict(minmax_df)
print(y_pred_hie.shape)
y_pred_hie


# In[1291]:


# Cluster numbers

agg_clustering.n_clusters_


# In[1292]:


# Clustering Score

(sil(minmax_df, agg_clustering.labels_)*100).round(3)


# In[1293]:


#Concating Labels with main dataset copy
df['Hierarchical_Labels'] = agg_clustering.labels_
df.groupby('Hierarchical_Labels').agg(['mean'])


# In[1294]:


#using PCA

from sklearn.decomposition import PCA


# In[1295]:


pca = PCA(n_components=2)


# In[1296]:


pca


# In[1297]:


#
pca_scaled = pca.fit_transform(std_df)


# In[1298]:


pca_scaled


# In[1299]:


plt.scatter(pca_scaled[:,0],pca_scaled[:,1])


# In[1310]:


import scipy.cluster.hierarchy as sc
plt.figure(figsize=(20,7))
plt.title("Dendrograms")

sc.dendrogram(sc.linkage(pca_scaled,method='ward'))
plt.title('dendrogram')

plt.ylabel('eucledian distance')


# In[1311]:


from sklearn.cluster import AgglomerativeClustering



# In[1312]:


#
agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
y_pred_hie = agg_clustering.fit_predict(pca_scaled)
print(y_pred_hie.shape)
y_pred_hie


# In[1313]:


#
(sil(minmax_df, agg_clustering.labels_)*100).round(3)


# In[1314]:


#DBSCAN
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[1315]:


from sklearn.cluster import DBSCAN


# In[1316]:


#
dbscan = DBSCAN(eps=2.5, min_samples=21)


# In[1317]:


dbscan.fit(std_df)


# In[1318]:


#
labels = dbscan.labels_


# In[1319]:


labels


# In[1320]:


#
df['dbscan'] = labels
print(df['dbscan'].value_counts())


# In[1321]:


#
df.groupby(df['dbscan']).agg(['mean'])


# In[1322]:


df.groupby(df['dbscan']).count()['ID'].plot(kind='bar')
plt.ylabel('ID Counts')
plt.title('DBSCAN Clustering Standard Scaled Data',fontsize='large',fontweight='bold')
ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')
ax.set_ylabel('ID counts', fontsize='large', fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[1323]:


# min max data


# In[1324]:


dbscan1 = DBSCAN(eps=1, min_samples=22) # min_samples = number of clumns * 3
dbscan1.fit(minmax_df)


# In[1325]:


labels1 = dbscan1.labels_


# In[1326]:


labels1


# In[1327]:


#
df['dbscan1'] = labels1
print(df['dbscan1'].value_counts())


# In[1328]:


#
df.groupby(df['dbscan1']).agg(['mean'])


# In[1329]:


df.groupby(df['dbscan1']).count()['ID'].plot(kind='bar')
plt.ylabel('ID Counts')
plt.title('DBSCAN Clustering Standard Scaled Data',fontsize='large',fontweight='bold')
ax.set_xlabel('Clusters', fontsize='large', fontweight='bold')
ax.set_ylabel('ID counts', fontsize='large', fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[1330]:


#
cluster1 = pd.DataFrame(df.loc[df.dbscan==0].mean(),columns= ['Cluster1_Avg'])
cluster2 = pd.DataFrame(df.loc[df.dbscan1==1].mean(),columns= ['Cluster2_Avg'])
avg_df = pd.concat([cluster1,cluster2],axis=1)
avg_df


# In[ ]:





# In[ ]:




