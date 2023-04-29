#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1, 1], [-1, -1], [0, 0.5], [0.1, 0.5], [0.2, 0.2], [0.9, 0.5]])
Y = np.array([1, -1, -1, -1, 1, 1])
w = np.array([1, 1])
b = 0
epoch = 1
converged = False
while not converged:
    print("Epoch ", epoch, ":")
    misclassified = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        y_hat = np.dot(w, x) + b
        if y_hat*y <= 0:
            w = w + y*x
            b = b + y
            misclassified += 1
            print("Sample ", i+1, ": x=", x, " y=", y, " y_hat=", y_hat, " Incorrect, Update w=", w, " b=", b)
        else:
            print("Sample ", i+1, ": x=", x, " y=", y, " y_hat=", y_hat, " Correct")
    if misclassified == 0:
        converged = True
    epoch += 1
print("\nFinal weight vector: w=", w, " b=", b)
print("Decision boundary: ", w[0], "x1 + ", w[1], "x2 + ", b, "= 0")
x = np.linspace(-1, 1, 100)
y = -w[0]/w[1]*x - b/w[1]
plt.scatter(X[:,0], X[:,1], c=Y)
plt.plot(x, y, '-g')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1, 1], [-1, -1], [0, 0.5], [0.1, 0.5], [0.2, 0.2], [0.9, 0.5]])
Y = np.array([1, -1, -1, -1, 1, 1])
w = np.array([1, 1])
b = 0
epoch = 1
converged = False
while not converged:
    print("Epoch ", epoch, ":")
    misclassified = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        y_hat = np.dot(w, x) + b
        if y_hat*y <= 0:
            w = w + y*x
            b = b + y
            misclassified += 1
            print("Sample ", i+1, ": x=", x, " y=", y, " y_hat=", y_hat, " Incorrect, Update w=", w, " b=", b)
        else:
            print("Sample ", i+1, ": x=", x, " y=", y, " y_hat=", y_hat, " Correct")
    if misclassified == 0:
        converged = True
    epoch += 1
print("\nFinal weight vector: w=", w, " b=", b)
print("Decision boundary: ", w[0], "x1 + ", w[1], "x2 + ", b, "= 0")
x = np.linspace(-1, 1, 100)
y = -w[0]/w[1]*x - b/w[1]
plt.scatter(X[:,0], X[:,1], c=Y)
plt.plot(x, y, '-g')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# In[ ]:




