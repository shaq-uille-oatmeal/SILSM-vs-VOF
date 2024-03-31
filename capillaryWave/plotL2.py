import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import CubicSpline
from scipy.integrate import simps

t = np.arange(0.000, 0.051, 0.001)
sigma = 3000
K = 2*np.pi
rho = 1

w0 = np.sqrt(sigma*(K**3)/2/rho)

eps = K**2/ w0
T = t*w0         

h_64 = np.zeros(51)
h_64[0] = 0.01

h_32 = np.zeros(51)
h_32[0] = 0.01

h_16 = np.zeros(51)
h_16[0] = 0.01
            
for i in range(1, 51):
    data = pd.read_csv('64x64RDF/postProcessing/interface_{}.csv'.format(i))
    y = data.iloc[:,1]
    x = data.iloc[:,0]
    
    for j in range(0,len(x)):
        if (x[j] == 0.5):
            h_64[i] = y[j]-0.5

for i in range(1, 51):
    data = pd.read_csv('32x32RDF/postProcessing/interface_{}.csv'.format(i))
    y = data.iloc[:,1]
    x = data.iloc[:,0]
    
    for j in range(0,len(x)):
        if (x[j] == 0.5):
            h_32[i] = y[j]-0.5

for i in range(1, 51):
    data = pd.read_csv('16x16RDF/postProcessing/interface_{}.csv'.format(i))
    y = data.iloc[:,1]
    x = data.iloc[:,0]
    
    for j in range(0,len(x)):
        if (x[j] == 0.5):
            h_16[i] = y[j]-0.5

fit_64 = CubicSpline(T,h_64)
fit_32 = CubicSpline(T,h_32)
fit_16 = CubicSpline(T,h_16)

T_ = np.linspace(0,25,1000)

## Theoretical Solution

from scipy.special import erfc

coeff = [1, - np.sqrt(eps*w0),- eps*w0,  np.power(eps*w0,1.5),w0**2 ]

roots = np.roots(coeff)
z1 = roots[0]
z2 = roots[1]
z3 = roots[2]
z4 = roots[3]

Z1 = (roots[1]-roots[0])*(roots[2]-roots[0])*(roots[3]-roots[0])
Z2 = (roots[2]-roots[1])*(roots[0]-roots[1])*(roots[3]-roots[1])
Z3 = (roots[1]-roots[2])*(roots[3]-roots[2])*(roots[0]-roots[2])
Z4 = (roots[0]-roots[3])*(roots[2]-roots[3])*(roots[1]-roots[3])

a1 = z1/Z1*(w0**2)/(z1**2 - eps*w0)*np.exp((z1**2 - eps*w0)*T_/w0)*erfc(z1*(T_/w0)**0.5)
a2 = z2/Z2*(w0**2)/(z2**2 - eps*w0)*np.exp((z2**2 - eps*w0)*T_/w0)*erfc(z2*(T_/w0)**0.5)
a3 = z3/Z3*(w0**2)/(z3**2 - eps*w0)*np.exp((z3**2 - eps*w0)*T_/w0)*erfc(z3*(T_/w0)**0.5)
a4 = z4/Z4*(w0**2)/(z4**2 - eps*w0)*np.exp((z4**2 - eps*w0)*T_/w0)*erfc(z4*(T_/w0)**0.5)

a = (a1+a2+a3+a4)*0.01


## Sanjids Data
df = pd.read_csv("Fig2.3_SILSM.dat",delimiter='\t')
t_ = df.iloc[:,0].to_numpy()
h_ = df.iloc[:,1].to_numpy()

gridSize = np.array([16, 32, 64])

L2_64 = (simps((fit_64(T_)-a)**2,T_))**0.5
L2_32 = (simps((fit_32(T_)-a)**2,T_))**0.5
L2_16 = (simps((fit_16(T_)-a)**2,T_))**0.5

L2_16_LSM = 0.09
L2_32_LSM = 0.02
L2_64_LSM = 0.004

pl.loglog(gridSize, [L2_16, L2_32, L2_64],color='red')
pl.loglog(gridSize,[L2_16_LSM, L2_32_LSM, L2_64_LSM],color='blue')
pl.legend(['interFlow (RDF)','LSM'])
pl.xlabel('No. of grid points per unit length')
pl.ylabel('L2-Norm')
pl.title('L2-Norm Comparison')
pl.savefig('L2Norm.png')
pl.show()
