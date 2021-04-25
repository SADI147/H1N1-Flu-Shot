# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:10:33 2020

@author: Sourav
"""

import pandas as pd
#import numpy as np

data_features = pd.read_csv('training_set_features.csv')
data_labels = pd.read_csv('training_set_labels.csv')

data = pd.concat([data_features, data_labels], axis = 1)
data = data.drop(['respondent_id', 'seasonal_vaccine',
                  'h1n1_vaccine', 'behavioral_antiviral_meds',
                  'behavioral_wash_hands', 'child_under_6_months',
                  'behavioral_face_mask'], axis = 1)

data = data.fillna(data.ffill())
data_copy = pd.concat([data_features, data_labels], axis = 1)

data['age_group'] = data['age_group'].astype('category')
data['education'] = data['education'].astype('category')
data['race'] = data['race'].astype('category')
data['sex'] = data['sex'].astype('category')
data['income_poverty'] = data['income_poverty'].astype('category')
data['marital_status'] = data['marital_status'].astype('category') 
data['rent_or_own'] = data['rent_or_own'].astype('category')
data['employment_status'] = data['employment_status'].astype('category')
data['hhs_geo_region'] = data['hhs_geo_region'].astype('category')
data['census_msa'] = data['census_msa'].astype('category')
data['employment_industry'] = data['employment_industry'].astype('category')
data['employment_occupation'] = data['employment_occupation'].astype('category')

data = pd.get_dummies(data, drop_first=True)

import statsmodels.api as sm
Y = pd.concat([data_copy['h1n1_vaccine'], data_copy['seasonal_vaccine']], axis = 1)
X = data
#X = sm.add_constant(X)

from sklearn.model_selection import train_test_split as tts
train_x, test_x, train_y, test_y = tts(X, Y, test_size = 0.2,
                                       random_state = 1234)


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(random_state=1, max_iter=2000, 
                    activation = 'tanh',
                    solver = 'sgd',
                    alpha = 0.01).fit(train_x, train_y)

clf.fit(train_x, train_y)

new = clf.predict(test_x)

from sklearn import metrics
metrics.roc_auc_score(test_y, new)

count_misclassified = (test_y != new).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(test_y, new)
print('Accuracy: {:.4f}'.format(accuracy))

#------------------------------------------------------------------------------
#NOW COMES TESTING
#------------------------------------------------------------------------------
data_predict = pd.read_csv('test_set_features.csv')
data_predict_copy = pd.read_csv('test_set_features.csv')
data_predict = data_predict.drop(['respondent_id', 'behavioral_antiviral_meds',
                  'behavioral_wash_hands', 'child_under_6_months',
                  'behavioral_face_mask'
                                    ], axis = 1)

data_predict = data_predict.fillna(data_predict.ffill())

data_predict['age_group'] = data_predict['age_group'].astype('category')
data_predict['education'] = data_predict['education'].astype('category')
data_predict['race'] = data_predict['race'].astype('category')
data_predict['sex'] = data_predict['sex'].astype('category')
data_predict['income_poverty'] = data_predict['income_poverty'].astype('category')
data_predict['marital_status'] = data_predict['marital_status'].astype('category') 
data_predict['rent_or_own'] = data_predict['rent_or_own'].astype('category')
data_predict['employment_status'] = data_predict['employment_status'].astype('category')
data_predict['hhs_geo_region'] = data_predict['hhs_geo_region'].astype('category')
data_predict['census_msa'] = data_predict['census_msa'].astype('category')
data_predict['employment_industry'] = data_predict['employment_industry'].astype('category')
data_predict['employment_occupation'] = data_predict['employment_occupation'].astype('category')

data_predict = pd.get_dummies(data_predict, drop_first = True)
data_predict = sm.add_constant(data_predict)

pred = clf.predict_proba(data_predict)

#-------------------------------------------------------------
#final_pred_h1n1 = []
#final_pred_seasonal = []
#
#for i in pred[0]:
#    final_pred_h1n1.append(max(i))
#    
#
#for j in pred[1]:
#    final_pred_seasonal.append(max(j))
#    
#final_predictions_seasonal = pd.DataFrame(final_pred_seasonal, 
#                                   columns = ['seasonal_vaccine'])
#
#
#final_pred_h1n1 = pd.DataFrame(final_pred_h1n1, 
#                                   columns = ['h1n1_vaccine'])

pred = pd.DataFrame(pred, columns = ['h1n1_vaccine', 'seasonal_vaccine'])

file = pd.concat([data_predict_copy['respondent_id'], pred], axis = 1)
    
file.dtypes
file.to_csv('outputs/Output_ovr_4.csv', index = False)