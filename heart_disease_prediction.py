#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:18:49 2019

@author: ramesh
"""

"""
In this machine learning project, I have collected the dataset from Kaggle
 (https://www.kaggle.com/ronitf/heart-disease-uci) and I will be using Machine Learning \
 to make predictions on whether a person is suffering from Heart Disease or not.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproducibility

pd.options.mode.chained_assignment = None  #hide any pandas warnings


"""

Import dataset
===============
The dataset is stored in the file dataset.csv. 
I'll use the pandas read_csv method to read the dataset.

"""
dt = pd.read_csv('heart.csv')
dt.head(10)
"""
    age: The person's age in years
    sex: The person's sex (1 = male, 0 = female)
    cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
    trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
    chol: The person's cholesterol measurement in mg/dl
    fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
    restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
    thalach: The person's maximum heart rate achieved
    exang: Exercise induced angina (1 = yes; 0 = no)
    oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here https://litfl.com/st-segment-ecg-library/)
    slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
    ca: The number of major vessels (0-3)
    thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
    target: Heart disease (0 = no, 1 = yes)

"""

"""
Diagnosis: The diagnosis of heart disease is done on a combination of clinical signs and test results. 
The types of tests run will be chosen on the basis of what the physician thinks is going on 1, 
ranging from electrocardiograms and cardiac computerized tomography (CT) scans,
 to blood tests and exercise stress tests 2.

Looking at information of heart disease risk factors led me to the following: high cholesterol, 
high blood pressure, diabetes, weight, family history and smoking 3. According to another source 4, 
the major factors that can't be changed are: increasing age, male gender and heredity. Note that thalassemia, 
one of the variables in this dataset, is heredity. Major factors that can be modified are: Smoking, high cholesterol,
 high blood pressure, physical inactivity, and being overweight and having diabetes. Other factors include stress, alcohol and poor diet/nutrition.

I can see no reference to the 'number of major vessels', but given that the definition of
 heart disease is "...what happens when your heart's blood supply is blocked or 
 interrupted by a build-up of fatty substances in the coronary arteries",
 it seems logical the more major vessels is a good thing, and therefore will reduce the probability of heart disease.

Given the above, I would hypothesis that, if the model has some predictive ability, we'll see these factors standing out as the most important.
"""




dt.columns = ['age', 
              'sex', 
              'chest_pain_type', 
              'resting_blood_pressure',
              'cholesterol',
              'fasting_blood_sugar', 
              'rest_ecg', 
              'max_heart_rate_achieved',
              'exercise_induced_angina',
              'st_depression', 
              'st_slope',
              'num_major_vessels', 
              'thalassemia', 
              'target']


dt.dtypes


"""
I'm also going to change the values of the categorical variables, to improve the interpretation later on,
"""



dt['sex'][dt['sex'] == 0] = 'female'
dt['sex'][dt['sex'] == 1] = 'male'

dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'typical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'atypical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'non-anginal pain'
dt['chest_pain_type'][dt['chest_pain_type'] == 4] = 'asymptomatic'

dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'
dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'
dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'

dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 0] = 'no'
dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 1] = 'yes'

dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'
dt['st_slope'][dt['st_slope'] == 2] = 'flat'
dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'

dt['thalassemia'][dt['thalassemia'] == 1] = 'normal'
dt['thalassemia'][dt['thalassemia'] == 2] = 'fixed defect'
dt['thalassemia'][dt['thalassemia'] == 3] = 'reversable defect'



dt.dtypes

"""
For the categorical varibles, we need to create dummy variables. I'm also going to drop the first category of each.
For example, rather than having 'male' and 'female', 
we'll have 'male' with values of 0 or 1 (1 being male, and 0 therefore being female).
"""

dt = pd.get_dummies(dt, drop_first=True)


dt.head()

"""

The Model

The next part fits a random forest model to the data,

"""





X_train, X_test, y_train, y_test = train_test_split(dt.drop('target', 1), dt['target'], test_size = .2, random_state=10) #split the data




model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)



estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values


#code from https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
#sudo apt install graphviz
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label='root',
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')


