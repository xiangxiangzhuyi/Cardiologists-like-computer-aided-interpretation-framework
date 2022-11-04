import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers,Sequential,Model
import sklearn.metrics as metrics
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
import os
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD,Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.io import loadmat
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
import random
import h5py


#CNN Model
class InputConv(tf.keras.layers.Layer):
    def __init__(self,filter_num,kernel_size=32,stride=2,name='inputconv',**kwargs):
        super(InputConv,self).__init__(name=name,**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,kernel_size=kernel_size,strides=stride)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu = tf.keras.layers.Activation('relu')

    def call(self,input):
        model1 = self.conv1(input)
        model1 = self.bn1(model1)
        model1 = self.relu(model1)
        
        return model1

class Convblock(tf.keras.layers.Layer):
    def __init__(self,filter_num,kernel_size=32,stride=2,Dropout_rate=0,name='convblock',**kwargs):
        super(Convblock,self).__init__(name=name,**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,kernel_size=kernel_size,strides=stride,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(Dropout_rate)
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,kernel_size=kernel_size,strides=1,padding='same')
        self.maxpool = tf.keras.layers.MaxPooling1D(strides=stride,padding='same')

    def call(self,input):
        model1 = self.conv1(input)
        model2 = self.maxpool(input)
        model1 = self.bn1(model1)
        model1 = self.relu(model1)
        model1 = self.dropout(model1)
        model1 = self.conv2(model1)
#         print(model1.shape,'-------------------------------------',input.shape,'-------------------------------------------',model2.shape)
        output = layers.add([model1,model2])
        return output

class Batchblock(tf.keras.layers.Layer):
    def __init__(self,filter_num,kernel_size=32,stride=2,Dropout_rate=0,name='batchblock',**kwargs):
        super(Batchblock,self).__init__(name=name,**kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,kernel_size=kernel_size,strides=stride,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2 = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(Dropout_rate)
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,kernel_size=kernel_size,strides=1,padding='same')
        self.maxpool = tf.keras.layers.MaxPool1D(strides=stride,padding='same')
    
    def call(self,input):
        model1 = self.bn1(input)
        model2 = self.maxpool(input)
        model1 = self.relu1(model1)
        model1 = self.conv1(model1)
        model1 = self.bn2(model1)
        model1 = self.relu2(model1)
        model1 = self.dropout(model1)
        model1 = self.conv2(model1)
#         print(model1.shape,'-------------------------------------',input.shape,'-------------------------------------------',model2.shape)
        output = layers.add([model1,model2])
        return output

def ECGdet_CNN(num_classes):

    input_signal = tf.keras.Input(shape=(5000,9),name='input_signal')
    Signal_block_conv = InputConv(filter_num=32,kernel_size=50,stride=5,name='signal_block_conv')(input_signal)
    start_conv = Convblock(filter_num=32,kernel_size=3,stride=5,Dropout_rate=0.1,name='start_conv')(Signal_block_conv)
    for i in range(15):
        if i==0:
            batchblock = Batchblock(filter_num=32,kernel_size=3,stride=1,Dropout_rate=0.1,name=('batchblock'+str(i)))(start_conv)
        else:
            batchblock = Batchblock(filter_num=32,kernel_size=3,stride=1,Dropout_rate=0.1,name=('batchblock'+str(i)))(batchblock)
       
    bn_layer = tf.keras.layers.BatchNormalization(axis=-1,name='bn_layer')(batchblock)
    relu_layer = tf.keras.layers.Activation('relu',name='relu_layer')(bn_layer)
    flat_layer = tf.keras.layers.Flatten(name='flat_layer')(relu_layer)
    Dense_layer1 = tf.keras.layers.Dense(128,activation='relu',name='dense_layer1')(flat_layer)
    Dense_layer2 = tf.keras.layers.Dense(num_classes,activation='softmax',name='dense_layer2')(Dense_layer1)
    
    model = tf.keras.models.Model(input_signal,Dense_layer2)
    return model

#LSTM model
def ECGdet_LSTM(num_classes):
    #步态1
    input_signal = tf.keras.Input(shape=(5000,9),name='input_signal')        
    Signal_block_lstm = tf.keras.layers.LSTM(units=1024,input_shape=(5000,9),name='signal_block_lstm')(input_signal) 

#     Dense_block_1 = tf.keras.layers.Dense(64, activation='relu',name='Dense_block_1')(Signal_block_lstm)
    Dense_block_2 = tf.keras.layers.Dense(num_classes, activation='softmax',name='Dense_block_2')(Signal_block_lstm)

    model = tf.keras.models.Model(input_signal,Dense_block_2)
    return model

#KNN
from sklearn.neighbors import KNeighborsClassifier
train_label_opt = train_label
test_label_opt = test_label
train_data_opt = train_fea_use
test_data_opt = test_fea_use
#网格搜索算法确定最优参数
param_digits =[{'weights':['uniform'],'n_neighbors':[i for i in range(1,11)]},{'weights':['distance'],'n_neighbors':[i for i in range(1,6)],'p':[i for i in range(1,6)]}]
knn_clf = KNeighborsClassifier(n_neighbors=18-i-1)
knn_clf.fit(train_data_opt,train_label_unique)
ypred = knn_clf.predict_proba(test_data_opt)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5000,random_state=1,n_jobs=-1)
forest.fit(train_data_opt,train_label_unique)
y_score = forest.predict_log_proba(test_data_opt)

#Xgboost
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
params={'booster':'gbtree',
        'objective': 'multi:softmax',
        'num_class':18-i-1,
        'eval_metric': 'auc',
        'eta': 0.1,
        'n_estimators': 20, 
        'gamma':0.2, 
        'max_depth': 10, 
        'min_child_weight': 2,
        'colsample_bytree': 1, 
        'colsample_bylevel': 1, 
        'subsample': 0.9, 
        'reg_lambda': 10, 
    #         'reg_lambda': 0, 
        'reg_alpha': 0.1,
    #         'reg_alpha': 0,
        'seed': 200,
        'nthread':8,
        'silent':1
    #         'scale_pos_weight':scale_pos_weight
    }
    watchlist = [(dtrain,'train')]
    bst=xgb.train(params,dtrain,num_boost_round=25,evals=watchlist)
    ypred=bst.predict(dtest)