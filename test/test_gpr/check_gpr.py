from smac.epm.gaussian_process.gradient_gpr import GaussianProcessRegressor
import numpy as np

#X = "1.         0.00160249 0.99400012 0.33381911 0.82068882 0.5332819 0.60041886 0.40351272 0.77964846 0.36028971 0.12720401 0.53741622 0.1113385  0.63192486"
#Y = "0.12337871 0.09661948 0.1231681  0.0931294  0.11673929 0.10414764 0.10736484 0.09727454 0.11510804 0.09474813 0.07793964 0.10435165 0.07654994 0.10880861"
#G = "1.03508766 -11.30785714 0.03520852 0.06234317 0.03923216 0.04948361 0.04650061 0.05697149 0.04036529 0.06012583 0.08711758 0.04929197 0.08839861 0.04526925"
X = "1.         0.00160249 0.99400012 0.33381912 0.82068882 0.5332819 0.60041886 0.40351272 0.77964846 0.36028971 0.12720401 0.53741622 0.1113385  0.63192486"
Y = "0.12337871 0.09661948 0.1231681  0.0931294  0.11673929 0.10414764 0.10736484 0.09727454 0.11510804 0.09474813 0.07793964 0.10435165 0.07654994 0.10880861"
G = "0.03508766 -11.30785714 0.03520852 0.06234317 0.03923216 0.04948361 0.04650061 0.05697149 0.04036529 0.06012583 0.08711758 0.04929197 0.08839861 0.04526925"

X = [float(x) for x in X.split()]
Y = [float(y) for y in Y.split()]
G = [float(g) for g in G.split(" ")]
#X = X[:5]
#Y = Y[:5]
#G = G[:5]

#g = GaussianProcessRegressor(use_octave=True)
g = GaussianProcessRegressor()
g.fit(np.array(X).reshape(-1, 1), np.array(Y), np.array(G).reshape(-1, 1))
X_test = np.array(range(1000)) / 1000 + 0.001
y_test = g.predict(X_test.reshape(-1, 1))