import pickle
import numpy as np
import support_read_data as srd
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

pa = srd.path + 'Model/model17_Bayesian5_final_experiment/save_results/'
pa_d = os.listdir(pa)

all_res1, all_res2 = [], []
for fo in pa_d:
    fu_fi = pa  + fo + '/res.txt'
    print(fo)
    with open(fu_fi, 'rb') as fi:
        res = pickle.load(fi)
    if len(res[0]) == 9:
        all_res1 += res
    else:
        all_res2 += res
# type
ty_str = ['macro', 'micro', 'weighted']

# analysis all result 1
all_mat1 = []
for li in all_res1:
    te_l, te_p = li[7], li[8]
    te_fl = np.argmax(te_p, -1)

    # f1 score
    mrt = []
    for ty in ty_str:
        f1_sc = f1_score(te_l, te_fl, average=ty)
        pr_sc = precision_score(te_l, te_fl, average=ty)
        re_sc = recall_score(te_l, te_fl, average=ty)
        mrt += [f1_sc, pr_sc, re_sc]

    # metric
    all_acc = accuracy_score(te_l, te_fl)
    cm = confusion_matrix(te_l, te_fl)
    each_acc = np.array([cm[x, x] / np.sum(cm[x, :]) for x in range(cm.shape[0])])
    mean_acc = np.mean(each_acc)

    # convert labels
    red_n = li[1]
    con_lab = np.array([16, 8, 5, 9, 11, 14, 15, 12, 0, 2, 10, 6, 3, 1, 4, 7, 17, 13])
    avi_ind = np.where(con_lab < 18 - red_n)[0]
    or_index = np.arange(18)
    new_index = [or_index[x] for x in avi_ind]
    new_each_acc = np.zeros(18)
    new_each_acc[new_index] = each_acc
    new_each_acc = new_each_acc[np.argsort(con_lab)]

    # append
    all_mat1.append(li[:5] + mrt + [all_acc, mean_acc] + list(new_each_acc) + [0 for x in range(18-len(new_each_acc))])
all_mat1 = np.array(all_mat1)

# analysis all result 2
all_mat2 = []
for li in all_res2:
    all_acc_li, each_acc_li, mrt_li = [], [], []
    for li1 in li[9]:
        te_l, te_p = li1[2], li1[3]
        te_fl = np.argmax(te_p, -1)

        # f1 score
        mrt = []
        for ty in ty_str:
            f1_sc = f1_score(te_l, te_fl, average=ty)
            pr_sc = precision_score(te_l, te_fl, average=ty)
            re_sc = recall_score(te_l, te_fl, average=ty)
            mrt += [f1_sc, pr_sc, re_sc]
        mrt_li.append(mrt)

        # metric
        all_acc = accuracy_score(te_l, te_fl)
        cm = confusion_matrix(te_l, te_fl)
        each_acc = np.array([cm[x, x] / np.sum(cm[x, :]) for x in range(cm.shape[0])])

        all_acc_li.append(all_acc)
        each_acc_li.append(each_acc)


    all_acc = np.mean(all_acc_li)
    mrt = np.mean(np.array(mrt_li), 0)
    each_acc = np.mean(np.stack(each_acc_li, 0), 0)
    mean_acc = np.mean(each_acc)

    # convert labels
    red_n = li[1]
    con_lab = np.array([16, 8, 5, 9, 11, 14, 15, 12, 0, 2, 10, 6, 3, 1, 4, 7, 17, 13])
    avi_ind = np.where(con_lab < 18 - red_n)[0]
    or_index = np.arange(18)
    new_index = [or_index[x] for x in avi_ind]
    new_each_acc = np.zeros(18)
    new_each_acc[new_index] = each_acc
    new_each_acc = new_each_acc[np.argsort(con_lab)]

    # append
    all_mat2.append(li[:5] + list(mrt) + [all_acc, mean_acc] + list(new_each_acc) + [0 for x in range(18-len(new_each_acc))])
all_mat2 = np.array(all_mat2)

# save the result
sa_pa = '/home/hulianting/Projects/Project08_ECG/data/result/'
np.save(sa_pa + 'result1.npy', all_mat1)
np.save(sa_pa + 'result2.npy', all_mat2)

# find out run unsuccessful
all_op = np.array([[z, y, x] for z in [3,4,5] for y in range(10) for x in [3,5,8,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]]).astype(np.uint8)
runed_op = all_mat2[:, [0, 1, 4]].astype(np.uint8)

all_op = [str(all_op[x,0]) + '_' + str(all_op[x,1]) + '_' + str(all_op[x,2]) for x in range(all_op.shape[0])]
runed_op = [str(runed_op[x,0]) + '_' + str(runed_op[x,1]) + '_' + str(runed_op[x,2]) for x in range(runed_op.shape[0])]
unrun_op = [x for x in all_op if x not in runed_op]
unrun_op = np.array([[int(y) for y in x.split('_')] for x in unrun_op])

