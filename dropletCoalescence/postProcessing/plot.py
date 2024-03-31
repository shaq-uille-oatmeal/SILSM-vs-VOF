import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.interpolate import PchipInterpolator

index = np.arange(0,61,1)

t = index*0.01

h = np.zeros(len(index))

for i in index:
	df = pd.read_csv("interface_{}.csv".format(i))
	y = df.iloc[:,1]
	h[i] = np.max(y) - 2.5

ds = pd.read_csv("Fig2.13_SILSM.csv")
ts = ds.iloc[:,0]
hs = ds.iloc[:,1]
cs = PchipInterpolator(ts,hs)
hs_int = cs(t)
pl.plot(t,h,color='green')
pl.plot(t,hs_int,color='blue')
pl.xlabel("Non-dimensional Time")
pl.ylabel("Non-dimensional Height")
pl.legend(["interFlow","SI-LSM"])
pl.savefig('DropletCoalescence.eps', format='eps')
pl.show()
	

