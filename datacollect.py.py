# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 00:17:44 2018

@author: nb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:15:26 2018

@author: nb
"""
from nltk.corpus import wordnet as wn
import pickle
import os 
# Make folder
if not os.path.isdir("./Datafolder"):
    os.mkdir("./Datafolder")
folderpath = "./Datafolder/Data/"
if not os.path.isdir(folderpath):
    os.mkdir(folderpath)
classpath = "./Datafolder/Class/"
if not os.path.isdir(classpath):
    os.mkdir(classpath)
# Make category
category_list=[["Adult"],["Arts","Entertainment","Game"],["Auto","Vehicle","Transport"],["Beauty","Fitness","Cosmetic","exercise"],["Book","Literature","Document"],["Business","Industry"],
               ["Computer","Electricity","Telecom","Communication"],["Finance","Cost","Money"],["Food","Drink"],["Medicine","Drug"],["Home","Garden","Building"],["Job","Education"],
               ["Law","Government"],["People","Society"],["Animal"],["Science"],["Sport"],["Travel","place"],["Biology"],["Culture","Religion"],["Plant","Tree","vegetable","Flower"]]
with open(classpath+"category.txt", "wb") as fp:
    pickle.dump(category_list, fp)
#Load category
with open(classpath+"category.txt", "rb") as fp:
    category = pickle.load(fp)
#Find all related words in WordNet
for words in category:
    temp=[]
    for word in words:
        print(word)
        data = wn.synsets(word,'n')
        temp.extend(set([w for dt in data for aa in dt.closure(lambda s:s.hyponyms()) for w in aa.lemma_names()]))        
        print(word)
    temp=set(temp)   
    with open(folderpath+words[0]+".txt","wb") as fp:
        pickle.dump(temp, fp)
    print(len(temp))
##read file        
#with open(folderpath+"Adult.txt", "rb") as fp:
#   xxx = pickle.load(fp)
#           
#print(len(xxx))
#for words in category:
#    with open(folderpath+words[0],"ab") as fp:
#        for word in words:
#            data = wn.synset(word+'.n.01')
#            dt = list(set([w for aa in data.closure(lambda s:s.hyponyms()) for w in aa.lemma_names()]))
#            pickle.dump(category_list, fp)
        
        