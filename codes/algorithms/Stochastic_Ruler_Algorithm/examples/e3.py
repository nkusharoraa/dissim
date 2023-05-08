import dissim
import numpy as np
def ex3(x):
  if x['x'] ==1:
    f= 0.3
  elif x['x']==2:
    f= 0.7
  elif x['x']==3:
    f = 0.9
  elif x['x']==4:
    f = 0.5
  elif x['x']==5:
    f = 1
  elif x['x']==6:
    f = 1.4
  elif x['x']==7:
    f = 0.7
  elif x['x']==8:
    f = 0.8
  elif x['x']==9:
    f = 0
  elif x['x']==10:
    f = 0.6
  return f+np.random.uniform(-0.5,0.5)

dom = {'x': [i for i in range(1,11)]}

sr_userdef2 = dissim.stochastic_ruler(dom,'user_defined', 1000, 'opt_sol', ex3)
print(sr_userdef2.optsol())