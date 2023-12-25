# sem3_q2.m
# Luc Berthouze, February 2018
from scipy.io import loadmat
from numpy import mean,std,cov,dot,diag,sqrt,real,max,zeros,where
from scipy.linalg import eig
import matplotlib.pyplot as plt

## Load data
data = loadmat('sem3_q2_data.mat')
data = data['data']

plt.figure(1)
plt.scatter(data[:,0],data[:,1]); # Produce a scatter plot in original feature space

# Normalisation
data_norm = (data-mean(data,0))/std(data,0); # Note vectorised operations for mean and standard deviation. NB: Unlike Matlab, python will divide by element... so no need for a element-wise operator. You need to specify along which axis (row or column) the mean and the standard deviation are calculated. mean(data) would return a single number (mean over all columns/rows)
plt.scatter(data_norm[:,0],data_norm[:,1],c='r'); # Normalised data in red

# Whitening
C = cov(data.T) # Need to transpose as matrix needs to be 2x2 (we are looking at the covariance of the features, not the covariance of the data samples). This is different from Matlab. NB: .T transpose in place (data does not change shape)
[lambda_arr,U] = eig(C) # Returns array (not diagonal matrix) of lambda (eigenvalues) and U (eigenvector matrix)
data_whitened = dot(diag(1./sqrt(real(lambda_arr))),dot(U.T,(data-mean(data,0)).T)) # When using ndarrays, the product is element-wise so to specify matrix product or matrix/vector product, you need to use the dot operator (from numpy). This makes the expression a bit unpleasant. The alternative is to use np.matrix, in which case the expression would look like this: 
# data_whitened = np.matrix(diag(1./sqrt(real(lambda_arr))))*np.matrix(U.T)*np.matrix(data-mean(data,0)).T
# The covariance matrix being symmetric and real, the eigenvalues should be real. 
plt.scatter(data_whitened.T[:,0],data_whitened.T[:,1],c='g'); # Whitened data in green

# PCA
# Two options here. One is to do things manually, the other is to use the Python function (see preprocessing package in sklearn). I'll provide code for the former. For the latter, please read the manual. 
C = cov(data_norm.T) # Calculate the covariance matrix of the normalised data
[lambda_arr,U] = eig(C) # Returns array (not diagonal matrix) of lambda (eigenvalues) and U (eigenvector matrix)
largest_eig_idx = where(lambda_arr==max(lambda_arr))[0] # Find index of largest eigenvalue
data_reduced = dot(data_norm,U[:,largest_eig_idx]); # This is a simple matrix vector multiplication (a dot product)
plt.scatter(data_reduced,zeros((len(data_reduced),1)),c='k'); # in black
plt.show()
