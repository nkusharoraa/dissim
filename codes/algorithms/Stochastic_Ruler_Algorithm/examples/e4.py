import dissim
import numpy as np
def facility_loc(x):
  #normal demand
  X1,Y1 = x['x1'],x['y1'] 
  X2,Y2 = x['x2'],x['y2'] 
  X3,Y3 = x['x3'],x['y3'] 
  avg_dist_daywise = []
  T0 = 30
  n = 6
  for t in range(T0):
      total_day = 0                  ##### total distance travelled by people
      ###### now finding nearest facility and saving total distance 
      #travelled in each entry of data
      for i in range(n):
          for j in range(n):
              demand=-1
              while(demand<0):    
                  demand = np.random.normal(180, 30, size=1)[0]
              total_day += demand*min(abs(X1-i)+abs(Y1-j) ,
                                      abs(X2-i)+abs(Y2-j),abs(X3-i)+abs(Y3-j) ) 
              ### total distance from i,j th location to nearest facility
      avg_dist_daywise.append(total_day/(n*n))    
  return sum(avg_dist_daywise)/T0

dom = {'x1' : [i for i in range(1,7)], 'y1':[i for i in range(1,7)],
       'x2' : [i for i in range(1,7)], 'y2':[i for i in range(1,7)],
       'x3' : [i for i in range(1,7)], 'y3':[i for i in range(1,7)]}
sr_userdef3 = dissim.stochastic_ruler(dom,'user_defined', 1000, 'opt_sol',facility_loc)
print(sr_userdef3.optsol())
