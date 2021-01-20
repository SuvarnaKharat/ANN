# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:18:43 2021

@author: mona
"""
# Data preprocessing

# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
Data = pd.read_csv("C:/Users/mona/Desktop/DATA SCIENCE/Deep Learning/Churn_Modelling.csv")
X = Data.iloc[:, 3:13]
Y = Data.iloc[:, 13]

# Create dummy variables
Geography = pd.get_dummies(X["Geography"],drop_first=True)
Gender = pd.get_dummies(X["Gender"],drop_first=True)

## Concatenate the dataframes

X = pd.concat([X,Geography,Gender],axis = 1)

## Drop unnecessary columns

X = X.drop(["Geography", "Gender"],axis = 1)

# Spliting the dataset into training set and test set


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# ANN 
# Importing keras libraries
import keras
from keras.models import Sequentials
from keras.layers import Dense
#from keras.layers import LeakyRelu , PRelu , ELU
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequentials()

# Adding input layer and first hidden layer
classifier.add(Dense(units = 6 , kernel_initializer = "he_uniform",activation="Relu",input_dim = 11))

#Adding the second hiddden layer
classifier.add(Dense(units = 6 , kernel_initializer = "he_uniform",activation="Relu"))

# Adding output layer
classifier.add(Dense(units = 1 , kernel_initializer = "glorot_uniform",activation="sigmoid"))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

# Fitting the ANN to the training set
model_history = classifier.fit(X_train,Y_train,validation_split=0.33, batch_size =10, nb_epoch = 100)

# List all data in history

print(model_history.history.keys())
#summerize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = upper left)
plt.show()


# Making predictions and evaluating the models
# Predicting the test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

#Making the confusion matrix
from sklearn.matrics import confusion_matrix
cm = confusion_matrix (Y_test,Y_pred)

# Calculate accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(Y_pred,Y_test)

