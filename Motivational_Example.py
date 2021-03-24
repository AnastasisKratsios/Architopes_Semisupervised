#!/usr/bin/env python
# coding: utf-8

# # Proof of Concept Demo

# In[12]:


# #------------------------------------#
# # Only For Motivational Example Only #
# #------------------------------------#
# ## Hyperparameters
# percentage_in_row = .5
# N = 50

# def f_1(x):
#     return x
# def f_2(x):
#     return x**2
# x_0 = 0
# x_end = 1

# test_size_ratio = 1


# ## Initializations

# In[13]:


import numpy as np
import pandas as pd
from random import uniform
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import tensorflow as tf
import sklearn.model_selection as sk


# In[16]:


# Initializations
## Counters & Related
n = int(round(percentage_in_row*N))
dt = x_end/N
x_current = x_0
i = 0
## Containers
x_1 = []
y_1 = []
x_2 = []
y_2 = []
## Build Pattern
while x_current < x_end:
    # Sample Current Point
    x = uniform(x_current, x_current+dt)
    # Check which pattern it belongs to
    if i % n <= round(n*percentage_in_row):
        x_1.append(x)
        y_1.append(f_1(x))
    else:
        x_2.append(x)
        y_2.append(f_2(x))
    # Update
    x_current = x_current + dt
    i = i+1

x = [*x_1, *x_2]
y = [*y_1, *y_2]


# In[17]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(x_1,y_1)
plt.scatter(x_2,y_2)


# # Split Training and Testing Datasets

# In[36]:


if test_size_ratio < 1:
    # Generate
    X_train, X_test, y_train, y_test = sk.train_test_split(x,
                                                           y,
                                                           test_size=test_size_ratio,
                                                           random_state=42)
    # Format
    X_train = pd.DataFrame({"X": X_train})
    X_test = pd.DataFrame({"X": X_test})
    data_y = pd.DataFrame({"X": y_train})
    data_y_test = pd.DataFrame({"X": y_test})
else:
    # Format
    X_train = pd.DataFrame({"X": x})
    X_test = pd.DataFrame({"X": x})
    data_y = pd.DataFrame({"X": y})
    data_y_test = pd.DataFrame({"X": y})
    y_train = data_y
    y_test = data_y_test

# Coersion
y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)


# ### Split Sub-Patterns

# In[30]:


# Split Datasets
X1_train, X1_test, y1_train, y1_test = sk.train_test_split(x_1,
                                                    y_1,
                                                    test_size=test_size_ratio,
                                                    random_state=42)
X2_train, X2_test, y2_train, y2_test = sk.train_test_split(x_2,
                                                    y_2,
                                                    test_size=test_size_ratio,
                                                    random_state=42)


# # Oracle Model:

# In[31]:


# # example of a model defined with the sequential api
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# # define the model
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# NN_model = Sequential()

# # The Input Layer :
# NN_model.add(Dense(100, kernel_initializer='normal',input_dim = 1, activation='relu'))

# # The Output Layer :
# NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# # Compile the network :
# NN_model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
# NN_model.summary()


# NN_model.fit(X1_train, y1_train, epochs=500, batch_size=32)


# ---
