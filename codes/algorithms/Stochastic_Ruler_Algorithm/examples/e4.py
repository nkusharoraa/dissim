import dissim
import numpy as np
def func2(x):
  x1,x2,x3,x4 = x[0],x[1], x[2],x[3]
  return (x1+10*x2)**2 + 5* (x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4 + 1 +np.random.normal(0,30)


dom = [[-100,100],[-100,100],[-100,100],[-100,100]]
init = [5,4,3,2]
func2AHA = dissim.AHA(func2,dom)
a = func2AHA.AHAalgolocal(100,4000,dom,init)
# print(a)
print(a[-1])
print(func2(a[-1]))