# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:25:27 2018

@author: MATTJE
"""

ALLdata.NormalizeVals('dir',dirNormType)
import GenCovMat as GCM
CovMat = GCM.GenCovMat(ALLdata)

########
#Attempts at plotting that don't really work
########

X = np.arange(0,CovMat.shape[0],1)
Y = np.arange(0,CovMat.shape[0],1)
Z_all = np.ones(CovMat.shape[0]*CovMat.shape[0])
X_all = np.ones(CovMat.shape[0]*CovMat.shape[0])
Y_all = np.ones(CovMat.shape[0]*CovMat.shape[0])
for i in X:
    for ii in Y:
        X_all[i*ALLdata.numFiles+ii] = i
        Y_all[i*ALLdata.numFiles+ii] = ii
        Z_all[i*ALLdata.numFiles+ii] = CovMat.iloc[i,ii]

Z = np.array(CovMat)
#X = np.arange(0,Z.shape[0],1)
#Y = np.arange(0,Z.shape[0],1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_xlabel('Station ID')
ax.set_ylabel('Station ID')
ax.set_zlabel('Covariance')


fig = plt.figure(figsize=(18, 18))
ax.contour3D(X_all, Y_all, Z_all, 50, cmap='viridis')
ax.scatter(X_all, Y_all, c=Z, cmap='viridis', linewidth=0.5);
ax.plot_surface(X_all, Y_all, Z_all, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Station ID')
ax.set_ylabel('Station ID')
ax.set_zlabel('Covariance')
plt.show()