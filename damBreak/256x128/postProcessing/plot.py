import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import CubicSpline

t = np.arange(0.000, 0.279, 0.005)

tc = 0.0763651272051663

T = t/tc      

h = np.zeros(len(t))
z = np.zeros(len(t))

for i in range(0, len(t)):
    data = pd.read_csv('interface_{}.csv'.format(i))
    y = data.iloc[:,1].to_numpy()
    x = data.iloc[:,0].to_numpy()
    
    h[i] = np.max(y)
    z[i] = np.max(x)

h = h / h[0]
z = z / z[0]

fit_h = CubicSpline(T,h)
fit_z = CubicSpline(T,z)

TexpZ = np.array([0.62, 0.8, 1.02, 1.19, 1.34, 1.5, 1.66, 1.81, 1.97])
Zexp = np.array([1.22, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78,  3])

TexpH = np.array([0.00, 0.8, 1.29, 1.74, 2.15])
Hexp = np.array([1.00, 0.89, 0.78, 0.67, 0.56])

df = pd.read_csv('ColumnHeightSanjid.csv')
t_lsm = df.iloc[:,0].to_numpy()
h_lsm = df.iloc[:,1].to_numpy()
fit_h_lsm = CubicSpline(t_lsm, h_lsm)

T_ = np.linspace(0,2.25,1000)
pl.plot(T_,fit_h_lsm(T_),color='green')
pl.plot(T_,fit_h(T_),color = 'blue')
pl.xlim([0, 2.25])
pl.ylim([0.2, 1.2])
pl.xlabel('T')
pl.ylabel('H')
pl.scatter(TexpH, Hexp, color = 'black')
pl.legend(['LSM','interFlow','Martin & Moyce, 1952'])
pl.title('Column Height')
pl.savefig('ColumnHeight.png')
pl.show()

df = pd.read_csv('LeadingEdgeSanjid.csv')
t_lsm_z = df.iloc[:,0].to_numpy()
z_lsm = df.iloc[:,1].to_numpy()
fit_z_lsm = CubicSpline(t_lsm_z, z_lsm)

pl.figure()
pl.plot(T_,fit_z_lsm(T_),color='green')
pl.plot(T_,fit_z(T_)*1.44/fit_z(0.8), color = 'blue')
pl.xlim([0, 2])
pl.ylim([0, 4])
pl.scatter(TexpZ, Zexp, color = 'black')
pl.legend(['LSM','interFlow','Martin & Moyce, 1952'])
pl.xlabel('T')
pl.ylabel('Z')
pl.title('Leading Edge')
pl.savefig('LeadingEdge.png')
pl.show()

