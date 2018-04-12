# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:55:50 2018

@author: nb
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
from sklearn import preprocessing
#############
class CATEGORY(object):
    def __init__(self,sess):
        self.batch_size=32
        self.loadData()                       # load data and preprocessing
        self.x_size = self.train_X.shape[1]   # length of each vector     
        self.y_size = self.train_y.shape[1]   # output classes
        self.build_modle()
#################################
    def shuff(self,x,y):
        idx = np.random.permutation(len(x))
        self.train_X, self.train_y = np.array(x)[idx], np.array(y)[idx]
#################################
    def loadData(self):
        #train data
        datapath="./Datafolder/Data/OK/"
        with open(datapath+"Train_Input.txt", "rb") as fp:
            x = pickle.load(fp)
        with open(datapath+"Train_label.txt", "rb") as fp:
            y = pickle.load(fp)   
            ##shuffle & split data
#        print(len(x))
#        print("!")
        xx = preprocessing.scale(x)            # scale to zero mean and unit variance
        self.shuff(xx,y)                       # data shuffle
        #test data
        with open(datapath+"Test_Input.txt", "rb") as fp:
            self.test_x = pickle.load(fp)
        with open(datapath+"Test_label.txt", "rb") as fp:
            self.test_y = pickle.load(fp) 
#################################           
    def add_layer(self,inputs, in_size, out_size, activation_function=None):
        with tf.name_scope('layer'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            ########## Batch Normalization  
            fc_mean, fc_var = tf.nn.moments(Wx_plus_b,axes=[0])  
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()
            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)  
            ##########       
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
        return outputs
#################################    
    def build_modle(self):
        with tf.name_scope('inputs'):
            self.xp = tf.placeholder("float", shape=[None, self.x_size], name='x_input')  
            self.yp = tf.placeholder("float", shape=[None, self.y_size], name='y_input')
        self.l1 = self.add_layer(self.xp, self.x_size, 1500, activation_function=tf.nn.relu)
        self.l2 = self.add_layer(self.l1, 1500, 1000, activation_function=tf.nn.relu)
        self.l3 = self.add_layer(self.l2, 1000, 500, activation_function=tf.nn.relu)
        self.l4 = self.add_layer(self.l3, 500, 150, activation_function=tf.nn.relu)
        self.l5 = self.add_layer(self.l4, 150, self.y_size, activation_function=None)  
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.yp, 1), tf.argmax(tf.nn.softmax(self.l5), 1)), tf.float32))
        with tf.name_scope('loss'):
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.yp, logits=self.l5))
            tf.summary.scalar('loss', self.loss)
#################################          
    def get_batch(self,flag,total,batch_size=50):
        tmp=total-flag
        if(tmp>=batch_size):
            lists=[flag+i for i in range(batch_size)]
        else:
            lists=[flag+i for i in  range(tmp)]
            lists2=[i for i in  range(batch_size-tmp)]
            lists.extend(lists2)
        return lists,(tmp<batch_size)
#################################          
    def train(self):
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(5e-4).minimize(self.loss)
        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        cnt=0
        for epoch in range(25):     
            for i in range(0,len(self.train_X),self.batch_size):
                lst,outornot=self.get_batch(i,len(self.train_X),self.batch_size)
                self.sess.run(self.train_step, feed_dict={self.xp: self.train_X[lst], self.yp: self.train_y[lst]})
#            print(i)
                if(i%(self.batch_size*10)==0):
                    rs = self.sess.run(self.merged,feed_dict={self.xp: self.train_X[lst], self.yp: self.train_y[lst]})
                    self.writer.add_summary(rs, cnt)
                cnt=cnt+self.batch_size
                if(outornot): break
            self.saver.save(self.sess, 'model/my_test_model',global_step=epoch)
            print("======")
            print("epoch" +str(epoch))
            print("Loss :"+str(self.sess.run(self.loss, feed_dict={self.xp: self.train_X, self.yp: self.train_y})))
            print("Training accuracy:  "+str(self.sess.run(self.accuracy, feed_dict={self.xp: self.train_X, self.yp: self.train_y})))
            self.shuff(self.train_X,self.train_y)
            print("Testing accuracy:  "+str(self.sess.run(self.accuracy, feed_dict={self.xp: self.test_x, self.yp: self.test_y}))) 
        self.sess.close()   
################################# 
def main():
    with tf.Session() as sess:
        CAT=CATEGORY(sess)
    CAT.train();
if __name__ == '__main__':
    main()