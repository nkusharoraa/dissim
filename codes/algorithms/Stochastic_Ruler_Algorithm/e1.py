import dissim
from sklearn import datasets

KNNspace ={ 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
KNNsr = dissim.stochastic_ruler(KNNspace , 'KNN' ,100)

digits = datasets.load_digits()
X_2 = digits.data[:, :-1]
y_2 = digits.target
result= (KNNsr.fit(X_2,y_2))
print(result)