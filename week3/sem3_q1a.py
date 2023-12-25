# sem3_q1a.py
# Luc Berthouze, February 2018
from numpy import linspace,arange
from numpy.random import normal
from scipy.stats import norm
import matplotlib.pyplot as plt

## Generate the data

# Generate 1D data from normal distribution
nbpts = 50 # Number of points to generate (50 in the brief, can be varied)
mu = 1 # the mean
sigma = 2 # the standard deviation

data = normal(mu,sigma,(nbpts,1)) # makes a column vector

# Create histogram. By default, this will show how many points fall in each bin. 
bins = 14 # This specifies the number of bins, Python decides the interval. This can be OK but means you do not specify exactly where the bins are. 
plt.hist(data, bins) 
plt.show()

# A different specification
edges = linspace(-5,8,14) # This specifies the number of bins and also where the first bin starts from
plt.hist(data, edges)
plt.show()

# Another specification
binsize = 1
edges = arange(-5,8,binsize) # This specifies a sequence defining the left edge of each bin (and, indirectly, the number of bins) 
plt.hist(data,edges)
plt.show()

## Normalized histogram
# We are using histograms to estimate probability densities. We need to remember that a probability density function has an area under the curve of 1 (when integrating over all values) so in order to be able to compare histogram and probability density function, we need to normalise our histograms. 
binsize = 1
edges = arange(-5,8,binsize) # varying the step size in the sequence means changing the number of bins. Consider what happens when changing the value of binsize
plt.hist(data, edges, normed=True)

# Now we can superimpose the pdf (probability density function) of the normal distribution we used to generate the data. We will plot it in the same range as covered by our edges

x = arange(-5,8,0.05) # the x values over which I want the PDF
y = norm.pdf(x,mu,sigma)
plt.plot(x,y,'r-') # plot with red line
plt.show()
x = -5:0.05:8; % the x values over which I want the PDF
y = normpdf(x,mu,sigma); 
plot(x,y,'r-'); % plot with red line.