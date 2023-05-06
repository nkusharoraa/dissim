import math as mt
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Union
import tracemalloc
import pandas as pd
import dask.dataframe as dd
import time
import random


#### Adaptive Hyperbox Algorithm

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
  


#### Stochastic Ruler Algorithm

def tracing_start():
    """
    Starts tracing of memory allocation using tracemalloc.
    """
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())
def tracing_mem():
    """
    Prints the peak memory usage.
    """
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak/(1024*1024)
    print("Peak Size in MB - ", peak)

class stochastic_ruler():
  """
  The class definition for the implementation of the Stochastic Ruler Random Search Method;
  Alrefaei, Mahmoud H., and Sigrun Andradottir. 
  "Discrete stochastic optimization via a modification of the stochastic ruler method."
  Proceedings Winter Simulation Conference. IEEE, 1996.
  """
  def __init__(self, space:dict,model_type:str,maxevals:int = 100, prob_type = 'hyp_opt', func=None):
    """The constructor for declaring the instance variables in the Stochastic Ruler Random Search Method 

    Args:
        space (dict): allowed set of values for the set of hyperparameters in the form of a dictionary
                      hyperparamater name -> key
                      the list of allowed values for the respective hyperparameter -> value
        model_type (str): string token for the worked model; from ['MLP':'MLPClassifier', 'SVM':'Support Vector Machine', 'RFC': Random Forest Classifier, 'DTC': Decision Tree Classifier, 'GBC' : Gradient Boosting Classifier ,'user_defined' : for an objective function different from the given models]
        maxevals (int, optional): maximum number of evaluations for the performance measure; Defaults to 100.
    """
    self.space = space
    self.prob_type = prob_type # hyperparam opt (hyp_opt), optimal solution (opt_sol)
    self.model_type = model_type #for already implemented models in scipy or userdefined
    self.data = None
    self.maxevals = maxevals
    self.initial_choice_HP = None
    self.Neigh_dict = self.help_neigh_struct()
    self.func = func

  def help_neigh_struct(self)->dict:
    """The helper method for creating a dictionary containing the position of the respective hyperparameter value in the enumered dictionary of space

    Returns:
        dict: hyperpamatername concatenated with its value ->key
              zero-based index position of the hyperparameter value in self.space (the hype) -> value
    """
    Dict = {}
    for hyper_param in self.space:
      for i,l in enumerate(self.space[hyper_param]):
        key = hyper_param + "_" + str(l)
        Dict[key] = i
    return Dict

  def random_pick_from_neighbourhood_structure(self, initial_choice:dict)->dict:
    """The method for picking a random hyperparameter set, in the form of a dictionary

    Args:
        initial_choice (dict): the initial choice of hyperparameters provided as a dictionary

    Returns:
        dict: a randomly picked set of hyperparameters written as a dictionary
    """

    set_hp = {}
    for hp in initial_choice:
      key = str(hp) + "_" + str(initial_choice[hp])
      hp_index = self.Neigh_dict[key]
      idx = np.random.randint(-1,2)
      length  = len(self.space[hp]) 
      set_hp[hp]=(self.space[hp][(hp_index+idx+length)%length])
    return set_hp


  def ModelFunc(self,neigh:dict):
    """ The function for calling the models from sklearn depending on the self.model_type string and the neighbourhood structure 

    Args:
        neigh (dict): the neighbourhood structure that needs to be traversed, provided in the form of a dictionary
                      hyperpamatername concatenated with its value ->key
                      zero-based index position of the hyperparameter value in self.space (the comprehensive set of values for the set of hyperparameters in the form of a dictionary)  -> value

    Returns:
        MODEL: the respective method called from sklearn with the chosen hyperparameters
    """
    if(self.model_type == 'MLP'):
      N = self.space['num_hidden_layers'][0]
      hidden_layer_sizes = []
      for i in range(N):
        hidden_layer_sizes.append(neigh['hidden_layer_'+str(i)])
      MODEL  = MLPClassifier(max_iter = 1000, solver = neigh['solver'], alpha = neigh['learning_rate'], hidden_layer_sizes = tuple(hidden_layer_sizes)) 
    if(self.model_type == 'SVM'):
      MODEL = SVC(kernel = neigh['kernel'],gamma = neigh['gamma'], C= neigh['C'],max_iter = 1000)
    if(self.model_type == 'RFC'):
      MODEL = RandomForestClassifier(max_depth=neigh['max_depth'],criterion =neigh['criterion'], n_estimators=neigh['n_estimators'])
    if(self.model_type == 'DTC'):
      MODEL = DecisionTreeClassifier(random_state=0)
    if(self.model_type == 'GBC'):
      MODEL = GradientBoostingClassifier(n_estimators=neigh['n_estimators'], learning_rate=neigh['learning_rate'],max_depth=neigh['max_depth'])
    if(self.model_type == 'KNN'):
      MODEL =  KNeighborsClassifier( n_neighbors =neigh['n_neighbors'],weights = neigh['weights'],metric= neigh['metric'])
    if(self.model_type == 'user_defined'):
      MODEL = self.func
    return MODEL

  
  def det_a_b(self,domain,max_eval,X = None,y =None):
    max_iter = (max_eval)//len(domain)
    maxm = -1*math.inf
    minm = math.inf
    neigh={}
    for i in range(max_iter):
      for dom in domain:
        neigh[dom] = np.random.choice(domain[dom])
      val = self.run(neigh,neigh, X, y)
      minm = min(minm,val)
      maxm = max(maxm,val)
    return (minm, maxm)


  def Mf(self,k:int)->int:
    """The method for represtenting the maximum number of failures allowed while iterating for the kth step in the Stochastic Ruler Method
        In this case, we let Mk = floor(log_5 (k + 10)) for all k; this choice of the sequence {Mk} satisfies the guidelines specified by Yan and Mukai (1992)
    Args:
        k (int): the iteration number in the Stochastic Ruler Method

    Returns:
        int: the maximum number for the number of iterations
    """
    return int(math.log(k+10,math.e)/math.log(5,math.e))

  def SR_Algo(self,X:np.ndarray = None,y:np.ndarray = None) -> Union[float,dict,float,float]:
    """The method that uses the Stochastic Ruler Method (Yan and Mukai 1992)
       Step 0: Select a starting point Xo E S and let k = 0.
       Step 1: Given xk = x, choose a candidate zk from N(x) randomly
       Step 2: Given zk = z, draw a sample h(z) from H(z). 
       Then draw a sample u from U(a, b). If h(z) > u, then let X(k+1) = Xk and go to Step 3. 
       Otherwise dmw another sample h(z) from H(z) and draw another sample u from U(a, b).
       If h(z) > u, then let X(k+1) = Xk and go to Step3. 
       Otherwise continue to draw and compare. 
       If all Mk tests, h(z) > u, fail, then accept the candidate zk and Set xk+1 = zk = Z. 

    Args:
        X (np.ndarray): feature matrix
        y (np.ndarray): label

    Returns:
        Union[float,dict]: the minimum value of 1-accuracy and the corresponding optimal hyperparameter set
    """
    # X_train,X_test,y_train,y_test = self.data_preprocessing(X,y)
    initial_choice_HP = {}
    for i in self.space:
      initial_choice_HP[i] = self.space[i][0]
    
    
    # step 0: Select a starting point x0 in S and let k = 0
    k = 1
    x_k = initial_choice_HP
    opt_x = x_k
    a, b = self.det_a_b(self.space,self.maxevals//10,X,y)
    # print(a,b)
    minh_of_z = b
    # step 0 ends here 
    while(k<self.maxevals+1):
      # print("minz"+str(minh_of_z))
      # step 1:  Given xk = x, choose a candidate zk from N(x)

      zk = self.random_pick_from_neighbourhood_structure(x_k)
      # step 1 ends here
      # step 2: Given zk = z, draw a sample h(z) from H(z)
      iter = self.Mf(k)
      for i in range(iter):
        h_of_z = self.run(zk,zk,X,y)
        
        u = np.random.uniform(a,b) # Then draw a sample u from U(a, b)
        
        if(h_of_z>u): # If h(z) > u, then let xk+1 = xk and go to step 3.
          k+=1 
          if(h_of_z<minh_of_z):
            minh_of_z = h_of_z
            opt_x = x_k
          break
        # Otherwise draw another sample h(z) from H(z) and draw another sample u from U(a, b), part of the loop where iter = self.Mf(k) tells the maximum number of failures allowed
      if(h_of_z<u): # If all Mk tests have failed
        x_k = zk
        k+=1 
        if(h_of_z<minh_of_z):
            minh_of_z = h_of_z
            opt_x = zk
      # step 2 ends here
      # step 3: k = k+1
    return minh_of_z,opt_x,a,b

  def data_preprocessing(self,X:np.ndarray,y:np.ndarray)->Union[list, list, list, list]:
    """The method for preprocessing and dividing the data into train and test using the train_test_split from sklearn,model_selection

    Args:
        X (np.ndarray): Feature Matrix in the form of numpy arrays
        y (np.ndarray): label in the form of numpy arrays

    Returns:
        Union[list, list, list, list]: returns X_train,X_test,y_train,y_test
    """
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train,X_test,y_train,y_test

  def fit(self,X:np.ndarray, y:np.ndarray)->Union[float,dict]: #for optimal hyperparams
    """The method to find the optimal set of hyperparameters employing Stochastic Ruler method

    Args:
        X (np.ndarray): Feature Matrix in the form of numpy arrays
        y (np.ndarray): label in the form of numpy arrays

    Returns:
        Union[float,dict]: the minimum value of 1-accuracy and the corresponding optimal hyperparameter set
    """
    tracing_start()
    start = time.time()
    
    self.data = {'X':X,'y':y}
    result = self.SR_Algo(X,y)
    end = time.time()
    print("time elapsed {} milli seconds".format((end-start)*1000))
    tracing_mem()
    return result
  
  def optsol(self):
    tracing_start()
    start = time.time()
    result = self.SR_Algo()
    
    end = time.time()
    print("time elapsed {} milli seconds".format((end-start)*1000))
    tracing_mem()
    return result

  def run(self, opt_x ,neigh:dict,X:np.ndarray = None ,y:np.ndarray = None)->float:
    """The (helper) method that instantiates the model function called from sklearn and returns the additive inverse of accuracy to be minimized

    Args:
        neigh (dict): helper dictionary with positions as values and the concatenated string of hyperparameter names and their values as their keys
        X (np.ndarray): Feature Matrix in the form of numpy arrays
        y (np.ndarray): label in the form of numpy arrays

    Returns:
        float: the additive inverse of accuracy to be minimized
    """
    ModelFun = self.ModelFunc(self.random_pick_from_neighbourhood_structure(neigh))
    if self.prob_type == 'hyp_opt':
      X_train,X_test,y_train,y_test = self.data_preprocessing(X,y)
      ModelFun.fit(X_train,y_train)
      y_pred = ModelFun.predict(X_test)
      acc = ModelFun.score(X_test,y_test)
      return 1-acc
    if self.prob_type== 'opt_sol':
      
      funcval = self.func(opt_x)
      
      # print(funcval,opt_x)
      return funcval
    # print("acc" + str(acc))
    