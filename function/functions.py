#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import sys



import sklearn.metrics as metrics

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings(action='ignore')


import os


# In[2]:


def random_forest(features,label,output_loc,bootstrap_num = 100,data_cut=0,feature_size=0):

    
    performance = np.zeros((bootstrap_num,))
    
    
    if feature_size:
        feature_cut = features.shape[1]-np.abs(feature_size)
    else:
        feature_cut = 0
    
    feature_importance = np.zeros((bootstrap_num,features.shape[1]-feature_cut))
    feature_importance_rank = np.zeros((bootstrap_num,features.shape[1]-feature_cut))
    
    
    for bootstrap in range(bootstrap_num):        
        if data_cut:
            
            features_cut,trash_a ,label_cut,trash_b  = train_test_split(features, label, train_size=(data_cut/label.shape[0]),shuffle=True,stratify=label,random_state=bootstrap)
            training_features, test_features, training_label, test_label = train_test_split(features_cut, label_cut, train_size=0.8,shuffle=True,stratify=label_cut,random_state=bootstrap)
 
        elif feature_size:
            if feature_size<0:
                training_features, test_features, training_label, test_label = train_test_split(features[:,features.shape[1]+feature_size:], label, train_size=0.8,shuffle=True,stratify=label,random_state=bootstrap)
            else:
                training_features, test_features, training_label, test_label = train_test_split(features[:,:feature_size], label, train_size=0.8,shuffle=True,stratify=label,random_state=bootstrap)

            
                
        else:   
            training_features, test_features, training_label, test_label = train_test_split(features, label, train_size=0.8,shuffle=True,stratify=label,random_state=bootstrap)
        X_train = pd.DataFrame(training_features)
        X_test = pd.DataFrame(test_features)
        y_train =  training_label
        y_test  = test_label 



        model_randomforest = RandomForestClassifier(random_state= 0)    
        model_randomforest.fit(X_train, y_train)
        predicted = model_randomforest.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test,predicted, pos_label=1)
        performance[bootstrap] = metrics.auc(fpr, tpr)
        
        feature_importance[bootstrap,:] = np.abs(model_randomforest.feature_importances_.reshape(-1,))
        feature_importance_rank[bootstrap,:] = features.shape[1]- feature_importance[bootstrap].argsort().argsort()
        # argsort.argsort는 각 feature의 위치에 해당 feautre의 순위에 반대되는, 즉, 1위는 feature size-1 크기를 배정해줌. 
        # features.shape[1]에서 뺌으로서, 제일 중요도가 큰 feature의 위치엔 1 이라는 숫자가 들어가게 됨. 
        # xth number in feature_importance_rank represents the rank of xth feature. The rank is the biggest to 1 and the smallest to feature_size. 
        
    if output_loc[-1]=="_":
        np.save(output_loc+'performance.npy',performance)
        np.save(output_loc+'feature_importance.npy',feature_importance)
        np.save(output_loc+'feature_importance_rank.npy',feature_importance_rank)
    else:
        np.save(output_loc+'/performance.npy',performance)
        np.save(output_loc+'/feature_importance.npy',feature_importance)
        np.save(output_loc+'/feature_importance_rank.npy',feature_importance_rank)
    return performance,feature_importance,feature_importance_rank


# In[3]:


def data_cut(data_name): 
    
    print("DATA CUT")
    print(data_name)
    features = np.load("../dataset/3.preprocessed/%s/features.npy"%data_name)
    label = np.load("../dataset/3.preprocessed/%s/label.npy"%data_name)
    
    performance_best = np.average(np.load("../result/%s/performance.npy"%data_name))
    
#     min_data_num = np.int(1/(np.sum(label)/label.shape[0]))+1
    
    
    
    
    if not 'data_cut' in os.listdir(os.getcwd()+"/../result/%s/"%data_name):
        os.mkdir("../result/%s/data_cut"%data_name)
        
    grid = np.array(([[label.shape[0],performance_best]]))
    np.save("../result/%s/data_cut/grid.npy"%data_name,grid)
#     else:
#         grid = np.load("../result/%s/data_cut/grid.npy"%data_name)
    
    current = 0 
    while(True):
        if grid.shape[0] ==current+1:
            if grid[current,1] <=0.55:
                print("STOP by average AUC under 0.55")
                break
#             elif grid[current,0] <min_data_num*2:
#                 break
            else:
                datasize = int(np.round(0.5*grid[current,0]))
                # what if AUC<0.55 by 0.5? 

                try:
                    performance,trash_a,trash_b = random_forest(features,label,"../result/%s/data_cut/%s_"%(data_name,datasize),100,datasize)  
                except:
                    print("AUC not available ")
                    break
                
                grid = np.append(grid,[[datasize,np.average(performance)]],axis=0)
                np.save("../result/%s/data_cut/grid.npy"%data_name,grid)
            
            
            
        if grid[current,1]-grid[current+1,1]<=0.05:
            current +=1
        else:
            
            if grid[current,0]-grid[current+1,0] ==1:
                current+=1
                continue
            
            datasize = int((grid[current,0]+grid[current+1,0])/2)
            performance,trash_a,trash_b = random_forest(features,label,"../result/%s/data_cut/%s_"%(data_name,datasize),100,datasize)    
            grid = np.append(np.append(grid[0:current+1,:].reshape(-1,2),
                                       np.array([[datasize,np.average(performance)]]),axis=0),
                             grid[current+1:,:].reshape(-1,2),axis=0)
            np.save("../result/%s/data_cut/grid.npy"%data_name,grid)
            
            
    return grid



def feature_cut(data_name): 
    
    print("FEATURE CUT")
    print(data_name)
    features = np.load("../dataset/3.preprocessed/%s/features.npy"%data_name)
    label = np.load("../dataset/3.preprocessed/%s/label.npy"%data_name)
    
    performance_best = np.average(np.load("../result/%s/performance.npy"%data_name))
    
    
    if not 'feature_cut' in os.listdir(os.getcwd()+"/../result/%s/"%data_name):
        os.mkdir("../result/%s/feature_cut"%data_name)
        
    grid = np.array(([[features.shape[1],performance_best]]))
    np.save("../result/%s/feature_cut/grid.npy"%data_name,grid)

    
    current = 0 
    while(True):
        # 
        if grid.shape[0] ==current+1:
            if grid[current,1] <=0.55:
                break
#             elif grid[current,0] <=features.shape[1]:
#                 break
            elif grid[current,0] <= 2:
                break

            else:
                featuresize = int(0.5*grid[current,0])                
                
                try:
                    performance,trash_a,trash_b = random_forest(features,label,"../result/%s/feature_cut/%s_"%(data_name,featuresize),100,0,-featuresize)   
                except:
                    print("AUC not available ")
                    break
                
#                 performance,trash_a,trash_b = random_forest(features,label,"../result/%s/feature_cut/%s_"%(data_name,featuresize),100,0,-featuresize)    
                grid = np.append(grid,[[featuresize,np.average(performance)]],axis=0)
                np.save("../result/%s/feature_cut/grid.npy"%data_name,grid)
            
            
        if grid[current,1]-grid[current+1,1]<=0.05:
            current +=1
        else:
            
            if abs(grid[current,0]-grid[current+1,0]) ==1:
                current+=1
                continue
            
            featuresize = int((grid[current,0]+grid[current+1,0])/2)
            performance,trash_a,trash_b = random_forest(features,label,"../result/%s/feature_cut/%s_"%(data_name,featuresize),100,0,-featuresize)    
            grid = np.append(np.append(grid[0:current+1,:].reshape(-1,2),
                                       np.array([[featuresize,np.average(performance)]]),axis=0),
                             grid[current+1:,:].reshape(-1,2),axis=0)
            np.save("../result/%s/feature_cut/grid.npy"%data_name,grid)
            
    return grid


# In[ ]:





# In[ ]:







# In[10]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




