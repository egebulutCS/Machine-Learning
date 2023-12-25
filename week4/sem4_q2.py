# sem4_q2.py
# Luc Berthouze, March 2018
from numpy import linspace,arange,zeros,concatenate
from numpy.random import normal
from scipy.stats import norm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## Generate the data

# Generate 1D data from bimodal distribution (two clusters) 
nbpts1 = 123 # Number of points in first cluster -- You can vary this point to see effect of having more or less points in one cluster
mu1 = -2 # the mean of the first mode (shift it left or right to increase separation between clusters)
sigma1 = 0.5 # the standard deviation of the first cluster -- vary this between tight (0.5) to large (2)
data1 = normal(mu1,sigma1,(nbpts1,1)) # makes a column vector
nbpts2 = 200 - nbpts1 # Number of points in second cluster
mu2 = 4 # the mean of the second mode (shift it left or right to increase separation between clusters)
sigma2 = 0.5 # the standard deviation of the second cluster -- vary this between tight (0.5) to large (2)
data2 = normal(mu2,sigma2,(nbpts2,1)) # makes a column vector
data = concatenate((data1,data2),axis=0) # Concatenate the data by row

## Normalized histogram. 
# Let's check that we get a unimodal density

binsize = 0.5
edges = arange(-10,15,binsize) # varying the step size in the sequence means changing the number of bins. Consider what happens when changing the value of binsize
plt.figure(100)
plt.hist(data, edges, normed=True)

# OK

## Apply K-Means with k taking values 5 to 1. 
# We want to see what happens when we apply k-means in the absence of prior
# knowledge as to how many clusters we can expect in the data. Here we know
# there is only one cluster but that's because we generated the data
# ourselves. 

for k in arange(5,0,-1):
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data) # Fit KMeans model with k cluster        
    idx = kmeans.predict(data) # Predict class membership
    plt.figure(k)
    plt.scatter(data[:, 0], zeros(len(data[:,0])), c=idx, s=50, cmap='viridis') # Borrowing style from tutorial
    C = kmeans.cluster_centers_
    plt.scatter(C[:, 0], zeros(len(C[:,0])), c='black', s=200, alpha=0.5); # Plotting centroids

plt.show()




