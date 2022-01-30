import numpy as np
import matplotlib.pyplot as plt
import giffilereader
from sklearn.preprocessing import StandardScaler
class PCA:
    def __init__(self):
        pass
    def apply(self):
        path = "Data\emotion_classification\\train2"
        scaler = StandardScaler()
        gfr = giffilereader.GIFFileReader()
        raw_data, happy_index, sad_index = gfr.readFile(path)
        #print("**************Raw Data***************\n",raw_data)
        raw_data_test = raw_data[:3,:3]
        #print("Shape of raw data\n",raw_data.shape)
        #print("Shape of test data\n",raw_data_test.shape)
        print("**************Test raw data***********\n", raw_data_test)
        raw_data_mean = raw_data_test.mean(axis = 0)
        print("**************Mean***************\n",raw_data_mean)
        N = raw_data.shape[0]

        X_centered = scaler.fit_transform(raw_data)

        #raw_data_centered = scaler.fit_transform(raw_data_test)
        raw_data_centered = scaler.fit(raw_data_test).transform(raw_data_test)
        print("**************Centered Data************\n",raw_data_centered)
        print("**************Centered data Mean***************\n",raw_data_centered.mean(axis=0))

        eigenvalues, eigenvectors = np.linalg.eigh((np.dot(X_centered, X_centered.T))/N)
        #print("**************Eigen values*************\n", eigenvalues)
        idx = eigenvalues.argsort()[::-1]
        #print("index =\t",idx)
        eigenvalues_v = eigenvalues[idx]
        eigenvectors_v = eigenvectors[:,idx]
        #print("**************Eigen values after sorting*************\n", eigenvalues_v)
        fig, ax = plt.subplots(3, 1, figsize=(12, 6))
        #plt.yscale("linear")
        #ax.set_yscale("log")
        n = np.array([0,1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        ax[0].plot(n, eigenvalues_v)
        ax[0].set_title("Eigen value plot")

        ax[0].set_yticks([0,500, 1000, 1500, 2000, 2500, 3000])
        ax[0].set_xticks(n)

        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2,2))
        ax[0].yaxis.set_major_formatter(formatter)
        ax[0].set_xlabel("i")
        ax[0].set_ylabel("eigen value")

        ax[1].set_title("Eigen value plot")
        ax[1].set_xticks(n)
        ax[1].set_yticks([0,1000, 2000, 3000])
        ax[1].yaxis.set_major_formatter(formatter)
        ax[1].set_xlabel("i")
        ax[1].set_ylabel("eigen value")
        ax[1].bar(n, eigenvalues_v, align="center", width=0.5, alpha=1)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=1)

        #print(eigenvectors_v.ndim)
        #print("**************Eigen vectors vi************\n", eigenvectors_v)
        eigenvectors_u = np.dot(X_centered.T, eigenvectors_v)
        eigenvectors_u = eigenvectors_u/np.linalg.norm(eigenvectors_u, axis=0)
        #print(eigenvectors_u.T.shape)
        #print("**************Eigen vectors ui************\n", eigenvectors_u)
        eigenvectors_d_dimension = eigenvectors_u[:, :2]
        #print(eigenvectors_d_dimension.T.shape)
        X_reduced = np.dot(eigenvectors_d_dimension.T, raw_data.T)
        return X_reduced.T, happy_index, sad_index
