import dissim
import numpy as np
def func5(x):
  xsum=0
  D = len(x)
  beta = 10000
  eps_star = 0
  gamma = .001
  for i in range(D):
    xsum += (x[i]-eps_star)**2
  
  return -1*beta*np.exp(-1*gamma*xsum)

m=10**20
D = 15
dom = [[np.ceil(-1*m**(1/D)/2),np.ceil(m**(1/D)/2)] for i in range(D)]
# print(dom)
init = [1 for i in range(D)]
func5AHA = dissim.AHA(func5, dom)
a= func5AHA.AHAalgolocal(100,10000,dom,init)
# print(x_star[-1],func5(x_star[-1]))
print(a[-1])
print(func5(a[-1]))