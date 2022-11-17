# HPTuningSRPackage
Implementation of the Stochastic Ruler Method for Discrete Simulation Optimization for Tuning ML Method Hyperparameters

## Documentation

### class stochastic_ruler():
  
  """
  The class definition for the implementation of the Stochastic Ruler Random Search Method;
  Alrefaei, Mahmoud H., and Sigrun Andradottir.
  "Discrete stochastic optimization via a modification of the stochastic ruler method."
  Proceedings Winter Simulation Conference. IEEE, 1996.
  """

 
  ### def __init__(self, space:dict,model_type:str,maxevals:int = 100):
    
    """The constructor for declaring the instance variables in the Stochastic Ruler Random Search Method
 
    Args:
        space (dict): allowed set of values for the set of hyperparameters in the form of a dictionary
                      hyperparamater name -> key
                      the list of allowed values for the respective hyperparameter -> value
        model_type (str): string token for the worked model; from ['MLP':'MLPClassifier', 'SVM':'Support Vector Machine']
        maxevals (int, optional): maximum number of evaluations for the performance measure; Defaults to 100.
    """

### def help_neigh_struct(self)->dict:
    
    """The helper method for creating a dictionary containing the position of the respective hyperparameter value in the enumerated dictionary of space
 
    Returns:
        dict: hyperpamater name concatenated with its value ->key
              zero-based index position of the hyperparameter value in self.space (the hype) -> value
    """
### def random_pick_from_neighbourhood_structure(self, initial_choice:dict)->dict:
    
    """The method for picking a random hyperparameter set, in the form of a dictionary
 
    Args:
        initial_choice (dict): the initial choice of hyperparameters provided as a dictionary
 
    Returns:
        dict: a randomly picked set of hyperparameters written as a dictionary
    """

### def ModelFunc(self,neigh:dict):
    
    """ The function for calling the models from sklearn depending on the self.model_type string and the neighbourhood structure
 
    Args:
        neigh (dict): the neighbourhood structure that needs to be traversed, provided in the form of a dictionary
                      Hyperpamater name concatenated with its value ->key
                      zero-based index position of the hyperparameter value in self.space (the comprehensive set of values for the set of hyperparameters in the form of a dictionary)  -> value
 
    Returns:
        MODEL: the respective method called from sklearn with the chosen hyperparameters
    """

### def Mf(self,k:int)->int:
    
    """The method for representing the maximum number of failures allowed while iterating for the kth step in the Stochastic Ruler Method
        In this case, we let Mk = floor(log_5 (k + 10)) for all k; this choice of the sequence {Mk} satisfies the guidelines specified by Yan and Mukai (1992)
    Args:
        k (int): the iteration number in the Stochastic Ruler Method
 
    Returns:
        int: the maximum number for the number of iterations
    """

### def SR_Algo(self,X:np.ndarray,y:np.ndarray) -> Union[float,dict]:
    
    """The method that uses the Stochastic Ruler Method (Yan and Mukai 1992)
       Step 0: Select a starting point Xo E S and let k = 0.
       Step 1: Given xk = x, choose a candidate zk from N(x) randomly
       Step 2: Given zk = z, draw a sample h(z) from H(z).
       Then draw a sample u from U(a, b). If h(z) > u, then let X(k+1) = Xk and go to Step 3.
       Otherwise draw another sample h(z) from H(z) and draw another sample u from U(a, b).
       If h(z) > u, then let X(k+1) = Xk and go to Step3.
       Otherwise continue to draw and compare.
       If all Mk tests, h(z) > u, fail, then accept the candidate zk and Set xk+1 = zk = Z.
 
    Args:
        X (np.ndarray): feature matrix
        y (np.ndarray): label
 
    Returns:
        Union[float,dict]: the minimum value of 1-accuracy and the corresponding optimal hyperparameter set
    """

### def data_preprocessing(self,X:np.ndarray,y:np.ndarray)->Union[list, list, list, list]:
    
    """The method for preprocessing and dividing the data into train and test using the train_test_split from sklearn, model_selection
 
    Args:
        X (np.ndarray): Feature Matrix in the form of numpy arrays
        y (np.ndarray): label in the form of numpy arrays
 
    Returns:
        Union[list, list, list, list]: returns X_train,X_test,y_train,y_test
    """

### def fit(self,X:np.ndarray,y:np.ndarray)->Union[float,dict]:
    
    """The method to find the optimal set of hyperparameters employing Stochastic Ruler method
 
    Args:
        X (np.ndarray): Feature Matrix in the form of numpy arrays
        y (np.ndarray): label in the form of numpy arrays
 
    Returns:
        Union[float,dict]: the minimum value of 1-accuracy and the corresponding optimal hyperparameter set
    """

  ### def run(self, neigh:dict,X:np.ndarray,y:np.ndarray)->float:
    
    """The (helper) method that instantiates the model function called from sklearn and returns the additive inverse of accuracy to be minimized
 
    Args:
        neigh (dict): helper dictionary with positions as values and the concatenated string of hyperparameter names and their values as their keys
        X (np.ndarray): Feature Matrix in the form of numpy arrays
        y (np.ndarray): label in the form of numpy arrays
 
    Returns:
        float: the additive inverse of accuracy to be minimized
    """
