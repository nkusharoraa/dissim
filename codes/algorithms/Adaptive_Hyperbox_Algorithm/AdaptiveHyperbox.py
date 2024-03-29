import math as mt
import numpy as np
import tracemalloc
import pandas as pd
import dask.dataframe as dd
import time
import random

def tracing_start():
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())
def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak/(1024*1024)
    print("Peak Size in MB - ", peak)
class AHA():
  def __init__(self, func,global_domain):
    """Constructor method that initializes the instance variables of the class AHA

    Args:
      func (function): A function that takes in a list of integers and returns a float value representing the objective function.
      global_domain (list of lists): A list of intervals (list of two integers) that defines the search space for the optimization problem.
    """
    self.func = func
    self.global_domain = global_domain
    # self.initial_choice = initial_choice
    self.x_star = []
    #self.initial_choice = self.sample(global_domain)
    
  def sample(self,domain):
    """A helper method that samples a point from the given domain. 
      It returns a list of integers representing a point in the search space.

    Args:
        domain (list of lists): A list of intervals (list of two integers) to sample from.

    Returns:
        list: A list of integers representing a point in the search space.
    """
    x = []
    for d in domain: #discrete domain from a to b
      a,b = d[0],d[1]
      x.append(np.random.randint(a,b+1))
    
    return x

  def ak(self,iters : int):
    """ A method that computes the value of the parameter 'a' for the given iteration.

    Args:
        iters (int): The current iteration number.

    Returns:
        int: The value of the parameter 'a'.
    """
    if iters <= 2.0: a = 3
    else: a = min(3, mt.ceil(3*(mt.log(iters))**1.01))
    return a

  #finding l_k u_k
  def hk_mpa(self,xbest_k,f,l_k,u_k):
    """Computes the values of 'l_k' and 'u_k' for the given iteration.

    Args:
        xbest_k (list): The current best solution found.
        f (list of lists): A list of unique solutions found so far.
        l_k (list): A list of lower bounds for each dimension in the search space.
        u_k (list): A list of upper bounds for each dimension in the search space.

    Returns:
        tuple: A tuple containing the updated values of 'l_k' and 'u_k'.
    """
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
    """Computes the set of candidate intervals 'c_k'.

    Args:
        l_k (list): A list of lower bounds for each dimension in the search space.
        u_k (list): A list of upper bounds for each dimension in the search space.

    Returns:
        list of lists: A list of intervals (list of two integers) representing the set of candidate intervals 'c_k'.
    """
    mpa_k = []
    for i in range(len(l_k)):
        mpa_k.append([l_k[i],u_k[i]])
    return mpa_k      



  def AHAalgolocal(self,max_k,m,loc_domain,x0):
    """ Runs the AHA algorithm for local optimization.

    Args:
        max_k (int): The maximum number of iterations to run.
        m (int): The number of random samples to generate at each iteration.
        loc_domain (list of lists): A list of intervals (list of two integers) that defines the local search space.
        x0 (list): A list of integers representing the initial solution.

    Returns:
        list of lists: A list of solutions found by the algorithm at each iteration.
    """
    #initialisation
    # print(x0)
    tracing_start()
    start = time.time()
    
    
    
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

    end = time.time()
    print("time elapsed {} milli seconds".format((end-start)*1000))
    tracing_mem()
    return self.x_star
  def AHAalgoglobal(self,iter,max_k,m):
    """ Runs the AHA algorithm for global optimization.

    Args:
        iter (int): The number of iterations to run.
    max_k (int): The maximum number of iterations to run for each iteration of the global search.
    m (int): The number of random samples to generate at each iteration.

    Returns:
        tuple: A tuple containing the best solution found and its corresponding objective value.
    """
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
