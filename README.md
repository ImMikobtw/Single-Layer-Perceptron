# Single-Layer Perceptron

## Topic
**Implementation of a Single-Layer Perceptron**

---

## Objective
To understand the principles of a **basic perceptron** — the simplest form of an artificial neural network — and to implement its learning algorithm from scratch using Python and NumPy.

---

## Theoretical Background
A **Perceptron** is a linear binary classifier that predicts whether an input belongs to one of two classes.

It consists of:
- **Input layer** — takes numerical features.
- **Weights** — determine the importance of each feature.
- **Activation function** — decides the output (0 or 1) based on a weighted sum.
- **Learning rule** — updates weights using the prediction error.

The perceptron learns by iteratively adjusting weights until the predictions match the expected outputs (for linearly separable data).

---

## 🧩 Implementation Details

### Step 1: Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
```

We use:
- **NumPy** for mathematical operations and array handling.
- **Matplotlib** (optional) for visualizing

### Step 2: Define the Perceptron Class
```python
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

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)
                z = np.dot(self.weights, xi)
                y_pred = self.activation(z)
                self.weights += self.lr * (target - y_pred) * xi
```
Let's explain:
- **weights**: includes one extra element for the bias term
- **activation()**: uses a **step function** -- outputs 1 if the value >= 0, else 0.
- **predict()**: computes the dot product of weights and inputs, then applies activation.
- **train()**: iteratively updates weights using the perceptron learning rule:
  
       wi​=wi​+η(y−y^​)xi​

where:
- η - learning rate
- y - actual label
- y^ - predicted label

### Step 3: Define the Dataset
```python
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 1])  # OR gate
```
This dataset represents the **logical OR operation**:
Input 1             Input 2              Output
0                   0                    0
0                   1                    1
1                   0                    1
1                   1                    1


### Step 4: Model Training
```python
p = Perceptron(input_size=2)
p.train(X, y, epochs=10)
```
Let's explain:
- This perceptron is trained for 10 epochs.
- In each epoch, it adjusts weights to reduce the prediction error.
- After training, the model learns to separate the classes with a linear boundary.

### Step 5: Testing the Model
```python
for i in range(len(X)):
    print(f"{X[i]} -> {p.predict(X[i])}")
```
Output should be:
```python
[0 0] -> 0
[0 1] -> 1
[1 0] -> 1
[1 1] -> 1
```

### Step 6: Visuals
You can visualize the decision boundary like that:
```python
x1 = np.linspace(-0.5, 1.5, 50)
x2 = -(p.weights[0] + p.weights[1]*x1) / p.weights[2]
plt.plot(x1, x2, label='Decision Boundary')
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Perceptron Decision Boundary')
plt.show()
```
