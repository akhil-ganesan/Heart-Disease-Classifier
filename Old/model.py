# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Importing Data
df =pd.read_csv("heart.csv")

# Converting Categorical to Numerical Categorical
from sklearn.preprocessing import LabelEncoder

x=df.drop("HeartDisease",axis=1)
y=df['HeartDisease']

for col in x.columns:
    x[col]=LabelEncoder().fit_transform(x[col])

from sklearn.model_selection import train_test_split

# Split so 80% training, 20% Testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,shuffle=True,random_state=0)

# Normalizing the data
from sklearn.preprocessing import StandardScaler

x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Training Logistic Regression
lr_model=LogisticRegression()
lr_model.fit(x_train,y_train)
y_pred_lr=lr_model.predict(x_test)

# Sensitivity and Specificity Analysis
con_lr=confusion_matrix(y_test,y_pred_lr)
# Run in terminal for heat map: 
#    import seaborn as sns 
#    sns.heatmap(con, annot=True, cmap='viridis', cbar=True)

# Accuracy Analysis
cr_lr=classification_report(y_test ,y_pred_lr)

from sklearn.neural_network import MLPClassifier

# Training MLP Classifier
nn_model=MLPClassifier(hidden_layer_sizes=[10, 10], activation='logistic', 
                       random_state=42, max_iter=300)
nn_model.fit(x_train,y_train)
y_pred_nn=lr_model.predict(x_test)

# Sensitivity and Specificity Analysis
con_mlp=confusion_matrix(y_test,y_pred_nn)
# Run in terminal for heat map: 
#    import seaborn as sns 
#    sns.heatmap(con, annot=True, cmap='viridis', cbar=True)

# Accuracy Analysis
cr_mlp=classification_report(y_test ,y_pred_nn)

# PCA Dimensionality Reduction then LR
from sklearn.decomposition import PCA

pca=PCA().fit(x)

dr_x_train=pca.transform(x_train)
dr_x_test=pca.transform(x_test)

# Retraining Logistic Regression
lr2_model=LogisticRegression()
lr2_model.fit(dr_x_train,y_train)
y_pred_lr2=lr2_model.predict(dr_x_test)

# Sensitivity and Specificity Analysis
con_lr2=confusion_matrix(y_test,y_pred_lr2)
# Run in terminal for heat map: 
#    import seaborn as sns 
#    sns.heatmap(con, annot=True, cmap='viridis', cbar=True)

# Accuracy Analysis
cr_lr2=classification_report(y_test ,y_pred_lr2)


