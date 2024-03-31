import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import CubicSpline

t = np.arange(0.000, 0.051, 0.001)
sigma = 3000
K = 2*np.pi
rho = 1

w0 = np.sqrt(sigma*(K**3)/2/rho)

eps = K**2/ w0
T = t*w0         
N = np.arange(0,51,1)

h_gradAlpha = np.zeros(51)
h_gradAlpha[0] = 0.01

h_RDF = np.zeros(51)
h_RDF[0] = 0.01

for i in range(1, 51):
    data = pd.read_csv('64x64GA/postProcessing/interface_{}.csv'.format(i))
    y = data.iloc[:,1]
    x = data.iloc[:,0]
    
    for j in range(0,len(x)):
        if (x[j] == 0.5):
            h_gradAlpha[i] = y[j]-0.5
            
for i in range(1, 51):
    data = pd.read_csv('64x64RDF/postProcessing/interface_{}.csv'.format(i))
    y = data.iloc[:,1]
    x = data.iloc[:,0]
    
    for j in range(0,len(x)):
        if (x[j] == 0.5):
            h_RDF[i] = y[j]-0.5


fit_gradAlpha = CubicSpline(T,h_gradAlpha)
fit_RDF = CubicSpline(T,h_RDF)
T_ = np.linspace(0,T[-1],1000)

## Theoretical Solution

from scipy.special import erfc

tau = np.linspace(0, 25, 1000)

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

a1 = z1/Z1*(w0**2)/(z1**2 - eps*w0)*np.exp((z1**2 - eps*w0)*tau/w0)*erfc(z1*(tau/w0)**0.5)
a2 = z2/Z2*(w0**2)/(z2**2 - eps*w0)*np.exp((z2**2 - eps*w0)*tau/w0)*erfc(z2*(tau/w0)**0.5)
a3 = z3/Z3*(w0**2)/(z3**2 - eps*w0)*np.exp((z3**2 - eps*w0)*tau/w0)*erfc(z3*(tau/w0)**0.5)
a4 = z4/Z4*(w0**2)/(z4**2 - eps*w0)*np.exp((z4**2 - eps*w0)*tau/w0)*erfc(z4*(tau/w0)**0.5)

a = a1+a2+a3+a4

pl.plot(tau,a*0.01,color='black',linestyle='dotted')

## Sanjids Data
df = pd.read_csv("Fig2.3_SILSM.dat",delimiter='\t')
t_ = df.iloc[:,0].to_numpy()
h_ = df.iloc[:,1].to_numpy()
pl.plot(t_,h_,color='green')

pl.plot(T_,fit_gradAlpha(T_),color='blue')
pl.plot(T_,fit_RDF(T_),color='red')
pl.legend(["theoretical solution",'LSM','interFlow (gradAlpha)', 'interFlow (RDF)'])
pl.xlim([0,25])
pl.title("Capillary Wave Amplitude")
pl.xlabel(r"$\tau$")
pl.ylabel("h")
pl.ylim([-0.015,0.015])





pl.savefig('capillarywave.png')
pl.show()
