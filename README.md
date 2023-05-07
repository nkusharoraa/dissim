# dissim - A discrete simulation optimization package

`dissim` is a Python package for discrete simulation optimization. It provides algorithms for solving optimization problems where the variables are discrete and the objective function is estimated through simulation. This can be useful in situations where the objective function is difficult to evaluate analytically or where the system being optimized has stochastic elements.

## Algorithms
 The package currently includes the following algorithms:

### Adaptive Hyperbox Algorithm
 This algorithm is an instance of a locally convergent, random search algorithm for solving discrete optimization via simulation problems. It is more efficient in high-dimensional problems compared to the COMPASS algorithm. This algorithm is described in the research paper “An adaptive hyperbox algorithm for high-dimensional discrete optimization via simulation problems” by Xu, Jie, Barry L. Nelson, and L. Jeff Hong.

### Stochastic Ruler Algorithm
 This algorithm was originally described by Yan and Mukai in 1992 for asymptotically determining the global optima of discrete simulation optimization problems. It has been proven to have asymptotic convergence in probability to the global optimum solution. This algorithm is described in the research paper “Stochastic discrete optimization” by Yan, Di, and H. Mukai.
 

## Requirements
 To use Dissim, you will need to have the following packages installed in your system:
  `numpy`
  `scikit-learn`
  `pandas`
  `dask`
 You can install these packages using pip by running `pip install numpy scikit-learn pandas dask`
## Installation 
 To install `dissim`, download the `dissim.whl` [file](https://github.com/nkusharoraa/dissim/raw/main/dissim.whl) from the root folder of the repository and run `pip install ./dissim.whl` in the same folder.
## Usage
 Example files for using the algorithms in this package can be found [here](https://github.com/nkusharoraa/dissim/tree/main/codes/algorithms/Adaptive_Hyperbox_Algorithm/examples) and [here](https://github.com/nkusharoraa/dissim/tree/main/codes/algorithms/Stochastic_Ruler_Algorithm/examples).
   
[Contact Us](mailto:nkusharoraa@gmail.com)