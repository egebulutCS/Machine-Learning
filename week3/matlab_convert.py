import scipy.io as spio

mat = spio.loadmat('sem3_q2_data.mat', squeeze_me=True)

print(mat)

a = mat['a'] # array
S = mat['S'] # structure containing an array
M = mat['M'] # array of structures

print(a[:,:])
print(S['b'][()][:,:]) # structures need [()]
print(M[0]['c'][()][:,:])
print(M[1]['c'][()][:,:])