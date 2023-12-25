# sem4_q1.py
# Luc Berthouze, March 2018
from numpy import linspace,arange,zeros
from numpy.random import normal
from scipy.stats import norm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## Generate the data
# copied from last week' seminar

# Generate 1D data from normal distribution (which is unimodal)
nbpts = 100 # Number of points to generate (100 in the brief, can be varied)
mu = 1 # the mean
sigma = 2 # the standard deviation

data = normal(mu,sigma,(nbpts,1)) # makes a column vector

## Normalized histogram. 
# Let's check that we get a unimodal density

binsize = 1
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