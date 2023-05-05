import math as mt
import numpy as np
class AHA():
  def __init__(self, func,global_domain):
    self.func = func
    self.global_domain = global_domain
    # self.initial_choice = initial_choice
    self.x_star = []
    #self.initial_choice = self.sample(global_domain)
    
  def sample(self,domain):
    x = []
    for d in domain: #discrete domain from a to b
      a,b = d[0],d[1]
      x.append(np.random.randint(a,b+1))
    
    return x

  def ak(self,itern):
      if itern <= 2.0: a = 3
      else: a = min(3, mt.ceil(3*(mt.log(itern))**1.01))
      return a

  #finding l_k u_k
  def hk_mpa(self,xbest_k,f,l_k,u_k):
      
      for i in range(len(xbest_k)):
          #f is the set of unique solutions
          for j in range(len(f)):
              if xbest_k != f[j] and l_k[i] <= f[j][i]:
                  if xbest_k[i] > f[j][i]:
                      l_k[i] = f[j][i]
              if xbest_k != f[j] and u_k[i] >= f[j][i]:
                  if xbest_k[i] < f[j][i]:
                      u_k[i] = f[j][i]
      return l_k, u_k

  def ck(self,l_k, u_k):
      mpa_k = []
      for i in range(len(l_k)):
          mpa_k.append([l_k[i],u_k[i]])
      return mpa_k      



  def AHAalgolocal(self,max_k,m,loc_domain,x0):
    #initialisation
    # print(x0)
    self.x_star.append(x0)
    epsilon = []
    epsilon.append([x0])
    G = 0
    for i in range(self.ak(0)):
      G += self.func(x0)
    G_bar_best = G/self.ak(0)

    l_k = []
    u_k = []
    for i in range(len(loc_domain)):
        l_k.append(loc_domain[i][0])
        u_k.append(loc_domain[i][1])

    # c = [domain]
    # phi = domain

    all_sol = [x0]
    uniq_sol_k =[]
    for k in range(1,max_k+1):
      all_sol_k = []
      
      xk =[]
      hk=[]

      l_k,u_k = self.hk_mpa(self.x_star[k-1],uniq_sol_k,l_k,u_k)
      ck_new = self.ck(l_k,u_k)  


      for i in range(m):
        xkm = self.sample(ck_new)
        # xk.append(xkm)
        all_sol_k.append(xkm)

      uniq_sol_k = list(set(tuple(x) for x in all_sol_k))
      # print(uniq_sol_k)
      all_sol.append(all_sol_k)
      # print(uniq_sol_k)
      epsilonk = uniq_sol_k + [tuple(self.x_star[k-1])]
      # print(epsilonk)
      x_star_k = self.x_star[k-1]
      for i in epsilonk:
        numsimobs = self.ak(k)
        g_val = 0
        for j in range(numsimobs):
          g_val += self.func(i)
        g_val_bar = g_val/numsimobs
        if(G_bar_best>g_val_bar):
          G_bar_best = g_val_bar
          x_star_k = list(i)
      self.x_star.append(x_star_k)
      epsilon.append(epsilonk)


    return self.x_star
  
  def AHAalgoglobal(self,iter,max_k,m):
      all_sols = []
      best_sol = None
      best_val = None

      # for i in range(iter):
      #   new_dom = []
      #   for dom in self.global_domain:
      #     if(dom[0]!=dom[1]):
      #       val1 = np.random.randint(dom[0],dom[1]+1)
      #       val2 = np.random.randint(dom[0],dom[1]+1)
      #       while(val1==val2):
      #         val2 = np.random.randint(dom[0],dom[1]+1)
      #       if(val1<val2):
      #         new_dom.append([val1,val2])
      #       else:
      #         new_dom.append([val2,val1])
      #     else:
      #       new_dom.append(dom)

      for i in range(iter):
        new_dom = []
        for dom in self.global_domain:
          D = (dom[1] - dom[0])
          if(D>=2 and D<=20):
            K=3
          elif(D>20 and D<=50):
            K=5
          elif(D>50 and D<=100):
            K = 10
          elif(D>100):
            K = 50



          if(dom[1]-dom[0]>2):
            step = (dom[1]-dom[0])//K
            val1 = np.random.randint(dom[0],dom[1]+1)
            if(val1+step+1<dom[1]):
              val2 = np.random.randint(val1+1,val1+step+1)
            else:
              val2 = np.random.randint(val1-step-1,val1)
            if(val1<val2):
              new_dom.append([val1,val2])
            else:
              new_dom.append([val2,val1])


          elif(dom[0]!=dom[1]):
            val1 = np.random.randint(dom[0],dom[1]+1)
            val2 = np.random.randint(dom[0],dom[1]+1)
            while(val1==val2):
              val2 = np.random.randint(dom[0],dom[1]+1)
            if(val1<val2):
              new_dom.append([val1,val2])
            else:
              new_dom.append([val2,val1])
          else:
            new_dom.append(dom)

        # print(new_dom)
        x0 = self.sample(new_dom)
        solx = self.AHAalgolocal(max_k,m,new_dom,x0)

        valx = self.func(solx[-1])
        if(best_sol == None or valx<best_val):
          best_sol = solx[-1]
          best_val = valx
        all_sols.append(solx[-1])

      return all_sols,best_sol,best_val
