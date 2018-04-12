# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 06:36:08 2018

@author: nb
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 05:14:04 2018

@author: nb
"""
import gensim
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split


def shuff(x,y):
    idx = np.random.permutation(len(x))
    x1,y1 = np.array(x)[idx], np.array(y)[idx]
    return x1,y1

#Load model
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 
SMOTE=False
#Load category
folderpath = "./Datafolder/Data/"
classpath = "./Datafolder/Class/"
outpath = "./Datafolder/Data/OK/"
with open(classpath+"category.txt", "rb") as fp:
    category = pickle.load(fp)
#Timer
import time
tStart = time.time()
#
xtrain=[]
ytrain=[]
for cnt,f_name in enumerate(category):
    with open(folderpath+f_name[0]+".txt","rb") as fp:
        tmp=pickle.load(fp)
        tmp1=[]
        for item in tmp:
            try:
                tmp1.append(model[item])
            except:
                pass
        ytrain.extend(cnt for i in range(len(tmp1)))
        xtrain.extend(tmp1)

xtrain, ytrain=shuff(xtrain,ytrain)
train_X, test_X, train_y, test_y=train_test_split(xtrain, ytrain, test_size=0.2, random_state=0)
###############################
if(SMOTE):
    from imblearn.over_sampling import SMOTE 
    sm = SMOTE(random_state=42)
    train_X, train_y = sm.fit_sample(train_X, train_y)

#
if not os.path.isdir(outpath):
    os.mkdir(outpath)
num_labels=len(np.unique(train_y))
all_Y=np.eye(num_labels)[train_y]    
with open(outpath+"Train_Input.txt","wb") as fp:
        pickle.dump(train_X, fp)
with open(outpath+"Train_Label.txt","wb") as fp:
        pickle.dump(all_Y, fp)
        
#print(len(train_X))
#print(len(all_Y))

#print(all_Y[0])

num_labelss=len(np.unique(ytrain))
yyy=np.eye(num_labelss)[test_y]  
with open(outpath+"Test_Input.txt","wb") as fp:
        pickle.dump(test_X, fp)
with open(outpath+"Test_Label.txt","wb") as fp:
        pickle.dump(yyy, fp)

tEnd = time.time()
print ("It cost %f sec" % (tEnd - tStart))#會自動做近位