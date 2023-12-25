import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

test = np.random.normal(0, 0.1, 1000)

fig, ax = plt.subplots(1, 1)

mean, var, skew, kurt = norm.stats(moments = 'mvsk')

x = np.linspace(norm.ppf(0.3), norm.ppf(0.7), 100)
ax.plot(x, norm.pdf(x,0,0.09), 'r-', lw=5, alpha=0.6, label='norm pdf')

rv = norm()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

vals = norm.ppf([0.001, 0.5, 0.999])
np.allclose([0.001, 0.5, 0.999], norm.cdf(vals))

#r = norm.rvs(size=1000)

ax.hist(test, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()