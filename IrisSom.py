from sklearn_som.som import SOM
from sklearn import datasets
from Estimator import Estimator

n_clusters = 3
iris = datasets.load_iris()
iris_data = iris.data[:, :3]
iris_label = iris.target

iris_som = SOM(m=n_clusters, n=1, dim=3)

print('iris_data')
print(iris_data)

iris_som.fit(iris_data, epochs=3)
predictions = iris_som.predict(iris_data)
print('iris_label')
print(iris_label.__class__)

print('predictions')
print(predictions.__class__)

estimation = Estimator.estimate(n_clusters, predictions, iris_label, iris_data)

print('estimation')
print(estimation[0])
