import dissim
import numpy as np
def multinodal(x):
  return (np.sin(0.05*np.pi*x)**6)/2**(2*((x-10)/80)**2)

def func1(x0):
  x1,x2 = x0[0],x0[1]
  return -(multinodal(x1)+multinodal(x2))+np.random.normal(0,0.3)

init = [0,0]
dom = [[0,100],[0,100]]
func1AHA = dissim.AHA(func1,dom)
a = func1AHA.AHAalgolocal(100,50,dom,init)
# print(b,c)
print(a[-1])
print(func1(a[-1]))