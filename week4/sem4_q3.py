# sem4_q3.py
# Luc Berthouze, March 2018
from numpy import linspace,arange,zeros,concatenate
from numpy.random import normal,multivariate_normal
from scipy.stats import norm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## Generate the data

# Generate 3 clusters of 2D data drawn from bivariate Gaussian distribution
# For the sake of generating more real-world data, I will use correlated data. 
# The function make_blobs in the tutorial is nice but doesn't give you enough control over the exact location and size/shape 
# The code below does. 

nbpts1 = 333 # Number of points in first cluster (vary to explore effect of number of points)
mu1 = [-2,4] # the mean of the first cluster (this is a vector as we are in 2D) -- move it around
sigma1 = [[0.5,0],[0,0.9]] # the covariance matrix of the first cluster -- this one is not correlated
data1 = multivariate_normal(mu1,sigma1,nbpts1) # 

nbpts2 = 333 # Number of points in second cluster (vary to explore effect of number of points)
mu2 = [1,-3] # the mean of the second cluster (this is a vector as we are in 2D) -- move it around
sigma2 = [[1.2,0.2],[0.2,2.3]] # the covariance matrix of the second cluster -- this one is correlated
data2 = multivariate_normal(mu2,sigma2,nbpts2) # 

nbpts3 = 1000-(nbpts1+nbpts2) # Number of points in 3rd cluster 
mu3 = [4,3] # the mean of the first cluster (this is a vector as we are in 2D) -- move it around
sigma3 = [[2.5,0.4],[0.4,1.3]] # the covariance matrix of the first cluster -- this one is correlated
data3 = multivariate_normal(mu3,sigma3,nbpts3) # 
data = concatenate((data1,data2,data3),axis=0) # Concatenate the data by row

## 2D plot of the data

# Let's plot the data in 2D to get a sense of the 3 clusters 
# Here I will plot each in a different colour so we know what the classes
# are. 

plt.figure(100);
plt.scatter(data1[:,0],data1[:,1], c='red', s=12)
plt.scatter(data2[:,0],data2[:,1], c='green', s=12)
plt.scatter(data3[:,0],data3[:,1], c='blue', s=12)

## Apply K-Means with k taking values 5 to 1. 
# We want to see what happens when we apply k-means in the absence of prior
# knowledge as to how many clusters we can expect in the data. Here we know
# there is only one cluster but that's because we generated the data
# ourselves. 

for k in arange(5,0,-1):
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data) # Fit KMeans model with k cluster        
    idx = kmeans.predict(data) # Predict class membership
    plt.figure(k)
    plt.scatter(data[:, 0], data[:,1], c=idx, s=50, cmap='viridis') # Borrowing style from tutorial
    C = kmeans.cluster_centers_
    plt.scatter(C[:, 0], C[:,1], c='black', s=200, alpha=0.5); # Plotting centroids

plt.show()




