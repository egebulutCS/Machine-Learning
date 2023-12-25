# sem3_q1b.m
# Luc Berthouze, February 2018
from numpy import linspace,arange,exp
from numpy.random import normal
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

## Generate the data

# Generate 1D data from normal distribution
nbpts = 50 # Number of points to generate (50 in the brief, can be varied)
mu = 1 # the mean
sigma = 2 # the standard deviation

data = normal(mu,sigma,(nbpts,1)) # makes a column vector

## Option 1: Use the kernel density estimator linked in the brief. 
# No code provided -- just read the manual.

## Option 2: Use the ksdensity function of Matlab
# Again read the manual but here is one option. 
# For comparison purposes I will include the histogram and the theoretical
# pdf (in thick dashed blue line)
binsize = 1 # bin size for the histogram
h = 0.5 # kernel window for kernal density estimation
edges = arange(-5,8,binsize) # varying the step size in the sequence means changing the number of bins. Consider what happens when changing the value of binsize
plt.hist(data, edges, normed=True)

pts = arange(-5,8,0.05) # the x values over which I want the PDF and the density estimation
y = norm.pdf(pts,mu,sigma)
plt.plot(pts,y,'b-.',linewidth=2.0) # plot with thick dashed blue line.

# We will use KernelDensity function from sklearn
# instantiate and fit the KDE model
h = 0.5 # Bandwidth of kernel
kde = KernelDensity(bandwidth=h, kernel='gaussian') # Set kernel to Gaussian
kde.fit(data) # Estimate the model based on the data
log_dens = kde.score_samples(pts.reshape(-1,1)) # Calculate the density estimates for desired data points (these return log densities)
plt.plot(pts,exp(log_dens),'g-') # plot estimated densities in green
plt.show()