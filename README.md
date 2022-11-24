# HPTuningSRPackage
Implementation of the Stochastic Ruler Method for Discrete Simulation Optimization for Tuning ML Method Hyperparameters

## Sample Implementation
```
import pandas as pd
df = pd.read_csv('data.csv')
df = df.drop([df.columns[-1],df.columns[0]],axis = 1)
df['result'] = df['diagnosis']
df = df.drop(['diagnosis'],axis = 1)
df = pd.get_dummies(df,columns = ['result'], drop_first = True)
df = df.rename(columns = {df.columns[-1]:'result'})
y = df['result'].to_numpy()
X = df.to_numpy()[:,:-1]
ANNspace = {'num_hidden_layers' : [1,2],
            'hidden_layer_0' : [2,4,6,8,10],
            'hidden_layer_1' : [1,2,3,4,5],
            'learning_rate' : [0.001,0.004,0.007,0.01],
            'solver' : ['adam'] }
SVMspace = {'kernel' : ["rbf","linear"],
            'gamma' : [0.001, 0.01, 0.1, 0.5, 1, 10, 30, 50, 80, 100],
            'C' : [0.01, 0.1, 1, 10, 100, 300, 500, 700, 800, 1000]}
RFCspace = {'max_depth' :[1,3,5,8], 
            'criterion':['gini', 'entropy'], 
            'n_estimators':[10,30,50,80,100]}
GBCspace = {'n_estimators':[10,30,50,80,100], 
            'learning_rate': [0.001,0.004,0.007,0.01], 
            'max_depth':[1,3,5,8]}
KNNspace ={ 'n_neighbors' : [5,7,9,11,13,15],
             'weights' : ['uniform','distance'],
             'metric' : ['minkowski','euclidean','manhattan']}

ANNsr = stochastic_ruler(ANNspace,'MLP',100)
SVMsr = stochastic_ruler(SVMspace,'SVM',100)
RFCsr = stochastic_ruler(RFCspace, 'RFC', 100)
GBCsr = stochastic_ruler(GBCspace, 'GBC',100)
KNNsr = stochastic_ruler(KNNspace , 'KNN' ,100)

z_vals1,opt_x1 = ANNsr.fit(X,y)
z_vals2,opt_x2 = SVMsr.fit(X,y)
z_vals3,opt_x3 = RFCsr.fit(X,y)
z_vals4,opt_x4 = GBCsr.fit(X,y)
z_vals5,opt_x5 = KNNsr.fit(X,y)
```
## Documentation
```
class stochastic_ruler():
  
  """
  The class definition for the implementation of the Stochastic Ruler Random Search Method;
  Alrefaei, Mahmoud H., and Sigrun Andradottir.
  "Discrete stochastic optimization via a modification of the stochastic ruler method."
  Proceedings Winter Simulation Conference. IEEE, 1996.
  """

 
  def __init__(self, space:dict,model_type:str,maxevals:int = 100):
    
    """The constructor for declaring the instance variables in the Stochastic Ruler Random Search Method
 
    Args:
        space (dict): allowed set of values for the set of hyperparameters in the form of a dictionary
                      hyperparamater name -> key
                      the list of allowed values for the respective hyperparameter -> value
        model_type (str): string token for the worked model; from ['MLP':'MLPClassifier', 'SVM':'Support Vector Machine']
        maxevals (int, optional): maximum number of evaluations for the performance measure; Defaults to 100.
    """

   def help_neigh_struct(self)->dict:
    
    """The helper method for creating a dictionary containing the position of the respective hyperparameter value in the enumerated dictionary of space
 
    Returns:
        dict: hyperpamater name concatenated with its value ->key
              zero-based index position of the hyperparameter value in self.space (the hype) -> value
    """
    def random_pick_from_neighbourhood_structure(self, initial_choice:dict)->dict:
    
    """The method for picking a random hyperparameter set, in the form of a dictionary
 
    Args:
        initial_choice (dict): the initial choice of hyperparameters provided as a dictionary
 
    Returns:
        dict: a randomly picked set of hyperparameters written as a dictionary
    """
    def ModelFunc(self,neigh:dict):
    
    """ The function for calling the models from sklearn depending on the self.model_type string and the neighbourhood structure
 
    Args:
        neigh (dict): the neighbourhood structure that needs to be traversed, provided in the form of a dictionary
                      Hyperpamater name concatenated with its value ->key
                      zero-based index position of the hyperparameter value in self.space (the comprehensive set of values for the set of hyperparameters in the form of a dictionary)  -> value
 
    Returns:
        MODEL: the respective method called from sklearn with the chosen hyperparameters
    """

  def Mf(self,k:int)->int:
    
    """The method for representing the maximum number of failures allowed while iterating for the kth step in the Stochastic Ruler Method
        In this case, we let Mk = floor(log_5 (k + 10)) for all k; this choice of the sequence {Mk} satisfies the guidelines specified by Yan and Mukai (1992)
    Args:
        k (int): the iteration number in the Stochastic Ruler Method
 
    Returns:
        int: the maximum number for the number of iterations
    """

  def SR_Algo(self,X:np.ndarray,y:np.ndarray) -> Union[float,dict]:
    
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

  def data_preprocessing(self,X:np.ndarray,y:np.ndarray)->Union[list, list, list, list]:
    
    """The method for preprocessing and dividing the data into train and test using the train_test_split from sklearn, model_selection
 
    Args:
        X (np.ndarray): Feature Matrix in the form of numpy arrays
        y (np.ndarray): label in the form of numpy arrays
 
    Returns:
        Union[list, list, list, list]: returns X_train,X_test,y_train,y_test
    """

  def fit(self,X:np.ndarray,y:np.ndarray)->Union[float,dict]:
    
    """The method to find the optimal set of hyperparameters employing Stochastic Ruler method
 
    Args:
        X (np.ndarray): Feature Matrix in the form of numpy arrays
        y (np.ndarray): label in the form of numpy arrays
 
    Returns:
        Union[float,dict]: the minimum value of 1-accuracy and the corresponding optimal hyperparameter set
    """

  def run(self, neigh:dict,X:np.ndarray,y:np.ndarray)->float:
    
    """The (helper) method that instantiates the model function called from sklearn and returns the additive inverse of accuracy to be minimized
 
    Args:
        neigh (dict): helper dictionary with positions as values and the concatenated string of hyperparameter names and their values as their keys
        X (np.ndarray): Feature Matrix in the form of numpy arrays
        y (np.ndarray): label in the form of numpy arrays
 
    Returns:
        float: the additive inverse of accuracy to be minimized
    """
```
   ### Modules needed: numpy, math, sklearn, typing
   
## Problem of Interest

A user in need of hyperparameter tuning, when searched in the docstring of the stochastic ruler class, the information about the input parameters needed in the constructor is clearly conveyed. The user then works with the development of a python dictionary with the hyperparameter names (referred to as HP throughout in the pictorial representation) as keys and lists as their values. 
The creation of an object gives access to the methods named above. The function call to fit gives us the optimal hyperparameter set and the corresponding accuracy value depending on the data and the model input as well.
Adequate Exceptions are thrown.

![Screenshot 2022-11-17 235657](https://user-images.githubusercontent.com/59050030/202527968-d711c962-11f6-4330-a340-7adb6a65fc5a.jpg)

## Instantiation and Function Calls
As the object is created and used to call the fit method, it first checks if the format is correct and then calls the SR_Algo method. With a return type a union of dictionary and float, the method first makes an initial choice of hyperparameters from the given search space and then the following steps followed (as described in the Stochastic Ruler Algorithm):
Step 0: Select a starting point Xo E S and let k = 0.
Step 1: Given xk = x, choose a candidate zk from N(x) randomly
Step 2: Given zk = z, draw a sample h(z) from H(z). Then draw a sample u from U(a, b). If h(z) > u, then let X(k+1) = Xk and go to Step 3.  Otherwise, draw another sample h(z) from H(z) and draw another sample u from U(a, b). If h(z) > u, then let X(k+1) = Xk and go to Step3. Otherwise, continue to draw and compare.
If all Mk tests, h(z) > u, fail, then accept the candidate zk and Set xk+1 = zk = Z.
Step3: k = k+1
Here, a call to Mk is made at every k, which has an integer return type (floored). Calls to the run function are made at every zk (neighbour candidate solution) which returns the 1 - accuracy value to solve the minimization problem. Comparison with the uniform random variable theta(0,1) is done using NumPy.

![Screenshot 2022-11-17 235832](https://user-images.githubusercontent.com/59050030/202528306-affd4508-0e4d-4be3-aed9-44518c21895c.jpg)

![Screenshot 2022-11-17 235859](https://user-images.githubusercontent.com/59050030/202528395-11570650-7a18-4399-9d15-8ff23bed5400.jpg)

![Screenshot 2022-11-17 235934](https://user-images.githubusercontent.com/59050030/202528545-790194d1-6ceb-48c8-8b32-c9f3dce19953.jpg)
