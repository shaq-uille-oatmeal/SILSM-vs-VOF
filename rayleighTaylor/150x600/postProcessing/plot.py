import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import CubicSpline

t = np.arange(0.000, 4.02, 0.1)
tc = 1
At = 0.5

T = t/tc      

spike = np.zeros(len(t))
bubble = np.zeros(len(t))

for i in range(0, len(t)):
    data = pd.read_csv('interface_{}.csv'.format(i))
    y = data.iloc[:,1].to_numpy()
    x = data.iloc[:,0].to_numpy()
    
    spike[i] = np.min(y)-2
    bubble[i] = np.max(y)-2

fit_spike = CubicSpline(T,spike)
fit_bubble = CubicSpline(T,bubble)

huangbubble = pd.read_csv("BubbleHuang.csv")
t_huang_bubble = huangbubble.iloc[:,0].to_numpy()
h_huang_bubble = huangbubble.iloc[:,1].to_numpy()

dlsm = pd.read_csv("Fig2.14SI-LSM_col-rise.csv")
t_lsm_bubble = dlsm.iloc[:,0].to_numpy()
h_lsm_bubble = dlsm.iloc[:,1].to_numpy()

T_ = np.linspace(0,4,1000)
pl.plot(T_*(At**0.5),fit_bubble(T_),color='blue')
pl.plot(t_lsm_bubble,h_lsm_bubble,color='green')
pl.scatter(t_huang_bubble,h_huang_bubble,color='black')
pl.legend(['interFlow','LSM','Huang et al. 2021'])
pl.xlim([0, 2.5])
pl.xlabel('T')
pl.ylabel('Bubble Tip Location')
pl.title('Rayleigh Taylor Instability - Bubble Tip Location')
pl.savefig('Bubble.png')
pl.show()

huangspike = pd.read_csv("SpikeHuang.csv")
t_huang_spike = huangspike.iloc[:,0].to_numpy()
h_huang_spike = huangspike.iloc[:,1].to_numpy()

dlsm = pd.read_csv("Fig2.14SI-LSM_col-fall.csv")
t_lsm_spike = dlsm.iloc[:,0].to_numpy()
h_lsm_spike = dlsm.iloc[:,1].to_numpy()

pl.plot(T_*(At**0.5),fit_spike(T_),color='blue')
pl.plot(t_lsm_spike,h_lsm_spike,color='green')
pl.scatter(t_huang_spike,h_huang_spike,color='black')
pl.legend(['interFlow','LSM','Huang et al. 2021'])
pl.xlim([0, 2.5])
pl.xlabel('T')
pl.ylabel('Spike Location')
pl.title('Rayleigh Taylor Instability - Spike Location')
pl.savefig('Spike.png')
pl.show()


