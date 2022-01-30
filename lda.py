import numpy as np
import matplotlib.pyplot as plt
import pca
from sklearn.preprocessing import StandardScaler

path = "Data\emotion_classification\\train2"

happy2 = []
p1 = pca.PCA()
X, happy_index, sad_index = p1.apply()
#print(X.shape)

colors = np.zeros(len(happy_index) + len(sad_index))
#print(colors)

for j in happy_index:
    colors[j] = 0
    happy2.append(X[j])
happy2 = np.array(happy2)
#print(happy2.shape)

sad2 = []
for j in sad_index:
    colors[j] = 1
    sad2.append(X[j])

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
color= ['red' if c == 0 else 'green' for c in colors]

ax[0].scatter(X[:, 0], X[:, 1], color=color)
ax[0].set_title("Data plotting")

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
ax[0].xaxis.set_major_formatter(formatter)
ax[0].yaxis.set_major_formatter(formatter)
ax[0].set_xlabel("Feature1")
ax[0].set_ylabel("Feature2")
sad2 = np.array(sad2)

N = X.shape[0]
scaler = StandardScaler()
#X = scaler.fit_transform(X)
#print(X)
#happy2_scalled = scaler.fit_transform(happy2)
#sad2_scalled = scaler.fit_transform(sad2)
covariance_T = np.dot(X.T, X)/N
#print(covariance_X)
#print(happy2)
happy_mean = np.array([happy2.mean(axis = 0)])
#print(happy_mean.shape)
sad_mean = np.array([sad2.mean(axis = 0)])
#print(sad_mean.shape)
#print(diff.shape)
covariance_B = np.dot((happy_mean - sad_mean).T, (happy_mean - sad_mean))
#print(covariance_B.shape)
covariance_W = covariance_T - covariance_B
#print(covariance_W)
#print("Inverse is ")
#print(np.linalg.inv(covariance_W))
#print("Sw inverse * Sb")
Swb = np.matmul(np.linalg.inv(covariance_W), covariance_B)
#print(Swb)
eigenvalues, eigenvectors = np.linalg.eigh(Swb)
idx = eigenvalues.argsort()[::-1]
#print("index =\t",idx)
eigenvalues_v = eigenvalues[idx]
#print(eigenvalues_v)
eigenvectors_v = eigenvectors[:,idx]
#print(eigenvectors_v.shape)
principal1D = np.array([eigenvectors_v[0]])
projection = np.dot(principal1D, X.T)
#print("Printing projection")
#print(projection)
#origin = [0, 0]
#plt.quiver(*origin, *principal1D.T, color=['b'], scale=21)
#plt.show()
