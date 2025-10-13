import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
  def __init__(self, input_size, learning_rate=0.1):
    self.weights = np.zeros(input_size + 1)
    self.lr = learning_rate

  def activation(self, x):
    return np.where(x >= 0, 1, 0)

  def predict(self, x):
    x = np.insert(x, 0, 1)
    z = np.dot(self.weights, x)
    return self.activation(z)

  def train(self, X, y, epochs = 10):
    for _ in range(epochs):
      for xi, target in zip(X, y):
        xi = np.insert(xi, 0, 1)
        z = np.dot(self.weights, xi)
        y_pred = self.activation(z)
        self.weights += self.lr * (target - y_pred) * xi
  
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 1])

p = Perceptron(input_size=2)
p.train(X, y, epochs=10)

for i in range(len(X)):
  print(f"{X[i]} -> {p.predict(X[i])}")
