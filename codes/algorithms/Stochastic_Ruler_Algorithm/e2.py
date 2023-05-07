import dissim
import numpy as np

def func(x0):
  x1,x2 = x0['x1'],x0['x2']
  def multinodal(x):
    return (np.sin(0.05*np.pi*x)**6)/2**(2*((x-10)/80)**2)

  return -(multinodal(x1)+multinodal(x2))+np.random.normal(0,0.3)

dom = {'x1' : [i for i in range(101)],'x2' : [i for i in range(101)]}
sr_userDef = dissim.stochastic_ruler(dom,'user_defined', 100000, 'opt_sol', func)
print(sr_userDef.optsol())
print(func({'x1':10,'x2':10}))