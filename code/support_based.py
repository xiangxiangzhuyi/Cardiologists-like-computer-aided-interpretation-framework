# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:43:45 2020

@author: pc
"""

import numpy as np
import pickle
import random
import string
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import interpolate

# set the global path ----------------------------------------
ba_pa = '/home/hulianting/Projects/Project08_ECG/'
mo_path = ba_pa + 'Model/model09_arrhythmia_primitives/'
da_path = ba_pa + 'data/LUDB/data/'

# view data
def view_sig(sig, ma=[0]):
    if len(ma) == 0:
        ma = np.zeros(len(sig)).astype(np.int32)
    figure(figsize=(30, 6), dpi=80)
    col = ['b', 'g', 'r', 'y', 'c', 'o']
    st, en, lab = 0, 1, col[ma[0]]
    for i in range(1, len(ma)):
        if ma[i] != ma[i-1] or i == len(ma)-1:
            en = i
            plt.plot(np.arange(st, en), sig[st: en], color=lab)
            st = i
            lab = col[ma[i]]
    plt.show()
    return

# Cubic Spline Interpolation
def Bspline(sig, tar_n):
    tck, u = interpolate.splprep([np.arange(len(sig)), sig], k=3, s=0)
    u = np.linspace(0, 1, tar_n, endpoint=True)
    out = interpolate.splev(u, tck)
    return out[1]

# convert to one hot images
def conv_one_hot(img, cal_num):
    mat_li = []
    for i in range(int(cal_num)):
        ma = img.copy()
        ma[ma == i] = 100
        ma[ma != 100] = 0
        ma[ma == 100] = 1
        mat_li.append(ma)
        
    return np.stack(mat_li, -1)

def com_mul_str(str_li):
    str_li = str_li + [''.join(random.sample(string.ascii_letters + string.digits, 4))]
    long_str = ''
    
    for s in str_li:
        long_str = long_str + '_' + str(s)

    return long_str[1: ]    

# prediction to mask
def pred_to_mask(pred_arr):
    max_ind = np.argmax(pred_arr, axis = -1)
    
    bb = []
    for i in range(pred_arr.shape[-1]):
        block = np.zeros(pred_arr.shape[0:-1])
        block[max_ind == i] = 1
        bb.append(block)
    cc = np.stack(bb, -1)
    
    return cc

# dice 1
def dice1(mask, pred):
    a = np.sum(mask*pred)*2
    b = np.sum(mask + pred)
    return a/b

# dice 2
def dice2(mask, pred):
    a = np.sum(mask*pred, axis = (1, 2, 3))*2
    b = np.sum(mask + pred, axis = (1, 2, 3))
    di_mat = a/b
    return np.mean(di_mat)

# dice 3
def dice3(mask, pred):
    di_mat = np.sum(mask*pred, axis = (0, 1, 2))*2/np.sum(mask + pred, axis = (0, 1, 2))
    return np.mean(di_mat)


def mean_dice(mask, pred):
    di_mat = np.sum(mask*pred, axis = (1, 2, 3))*2/(np.sum(mask + pred, axis = (1, 2, 3)) + 1e-10)
    s_n = np.sum(mask, axis = (1, 2, 3))
    s_n[s_n> 0] = 1
    sam_dice = np.sum(di_mat, axis = 1)/np.sum(s_n, axis = 1)
    cat_dice = np.sum(di_mat, axis = 0)/np.sum(s_n, axis = 0)
    all_dice = np.mean(cat_dice)
    
    return di_mat, sam_dice, cat_dice, all_dice

# save the model result
def save_result(result, strs, fi):
    # create folder
    folder_name = mo_path + 'save_results/' + strs + '/'
    if  not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    # file name list
    np.save(folder_name + fi, result)
        
    return

# read the model result
def read_result(folder, file):
    # folder name
    file_full_name = mo_path + 'save_results/' + folder + '/' + file
    if os.path.isfile(file_full_name):
        data = np.load(file_full_name, encoding='bytes', allow_pickle=True)
    else:
        data = 'nofile'
    
    return data

# save list
def save_list(result, strs, fi):
    folder_name = mo_path + 'save_results/' + strs + '/'
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    # file name list
    with open(folder_name + fi, 'wb') as file:
        pickle.dump(result, file)
        
    return

# read list
def read_list(strs, fi):
    file_full_name = mo_path + 'save_results/' + strs + '/' + fi
    if os.path.isfile(file_full_name):
        with open(file_full_name, 'rb') as file:
            res = pickle.load(file)
    else:
        res = 'nofile'
    return res

# read the prediction result
def read_pred(folder_name):
    folder_fullname = mo_path + 'save_results/' + folder_name + '/te_pred/'
    img_li = []
    mask_li = []
    pred_li = []
    for i in range(2000):
        # file full names
        img_name = folder_fullname + str(i) + '_img.npy'
        mask_name = folder_fullname + str(i) + '_mask.npy'
        pred_name = folder_fullname + str(i) + '_pred.npy'
        if not os.path.isfile(img_name):
            break
        # read
        img_li.append(np.load(img_name))
        mask_li.append(np.load(mask_name))
        pred_li.append(np.load(pred_name))
    
    return np.stack(img_li, 0), np.stack(mask_li, 0), np.stack(pred_li, 0)