import pickle
import numpy as np
import scipy.io as si
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import support_read_data as srd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

pa = '/home/hulianting/Projects/Project08_ECG/data/other_files/'
class_na = ['Normal','IST','SBR','IV block','AFL','SA','VT','PJC','AF','AV block','AT','AVJER','AVJE','PVCs','AVJT','VER','PACs','SA block']
pa_d = os.listdir(pa)
all_met = []
for fi in pa_d:
    cm = si.loadmat(pa + fi)[fi.split('.')[0]]
    # convert
    con_lab = np.array([16, 8, 5, 9, 11, 14, 15, 12, 0, 2, 10, 6, 3, 1, 4, 7, 17, 13])
    ind = np.argsort(con_lab)
    cm = cm[ind, :][:, ind]

    lab, pred = [], []
    for i in range(18):
        for j in range(18):
            lab += [i for x in range(cm[i,j])]
            pred += [j for x in range(cm[i,j])]

    f1_sc = f1_score(lab, pred, average='macro')
    pr_sc = precision_score(lab, pred, average='macro')
    re_sc = recall_score(lab, pred, average='macro')
    each_recall = np.array([cm[x, x] / np.sum(cm[x, :]) for x in range(cm.shape[0])])
    all_met.append([f1_sc, pr_sc, re_sc] + list(each_recall))

# read the best performance
pa = srd.path + 'Model/model17_Bayesian5_final_experiment/save_results/'
pa_d = os.listdir(pa)

all_res1, all_res2 = [], []
for fo in pa_d:
    fu_fi = pa  + fo + '/res.txt'
    with open(fu_fi, 'rb') as fi:
        res = pickle.load(fi)
    if len(res[0]) == 9:
        all_res1 += res
    else:
        all_res2 += res
# type
ty_str = ['macro', 'micro', 'weighted']

# analysis all result 1
all_mat = []
for li in all_res1:
    te_l, te_p = li[7], li[8]
    te_fl = np.argmax(te_p, -1)

    # f1 score
    mrt = []
    for ty in ty_str:
        f1_sc = f1_score(te_l, te_fl, average=ty)
        f1_sc = precision_score(te_l, te_fl, average=ty)
        re_sc = recall_score(te_l, te_fl, average=ty)
        mrt += [f1_sc, f1_sc, re_sc]

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
    all_mat.append(li[:5] + mrt + [all_acc, mean_acc] + list(new_each_acc) + [0 for x in range(18-len(new_each_acc))])
all_mat = np.array(all_mat)

ind = np.where(all_mat[:, 1] == 0)[0]
i_mat = all_mat[ind, 7]
max_ind = ind[np.argmax(i_mat)]
lab = all_res1[max_ind][7]
pred = all_res1[max_ind][8]
pr_lab = np.argmax(pred, -1)
cm = confusion_matrix(lab, pr_lab)

# convert
con_lab = np.array([16, 8, 5, 9, 11, 14, 15, 12, 0, 2, 10, 6, 3, 1, 4, 7, 17, 13])
avi_ind = np.where(con_lab < 18 - 0)[0]
ind = np.argsort(con_lab[avi_ind])
cm = cm[ind, :][:, ind]

lab, pred = [], []
for i in range(18):
    for j in range(18):
        lab += [i for x in range(cm[i, j])]
        pred += [j for x in range(cm[i, j])]

f1_sc = f1_score(lab, pred, average='macro')
pr_sc = precision_score(lab, pred, average='macro')
re_sc = recall_score(lab, pred, average='macro')
each_recall = np.array([cm[x, x] / np.sum(cm[x, :]) for x in range(cm.shape[0])])
all_met.append([f1_sc, pr_sc, re_sc] + list(each_recall))

met_li = np.concatenate(all_met)
met_na = ['Macro-F1 score', 'Macro-precision', 'Macro-recall'] + class_na
met_na = met_na + met_na + met_na
ty_li = ['LSTM' for y in range(21)] + ['1D CNN' for y in range(21)] + ['Our model' for y in range(21)]
da = [[met_li[x], met_na[x], ty_li[x]]for x in range(63)]
df = pd.DataFrame(da, columns = ['Value', 'Metrics', 'Models'])


plt.figure(figsize=(20, 10))
ax = sns.barplot(x="Metrics", y="Value", hue="Models", data=df)
plt.savefig('/home/hulianting/Projects/Project08_ECG/figures/compare/compare.svg')

'''
# draw
plt.figure(figsize=(20, 20))
sns.set(font_scale=1.6)
ax = sns.heatmap(cm, annot=True, fmt="d", cbar=False)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xticklabels(class_na)
ax.set_yticklabels(class_na)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
plt.yticks(rotation=0)
#plt.show()
plt.savefig('/home/hulianting/Projects/Project08_ECG/figures/optimization/' + fi.split('.')[0] + '  CM.svg')
'''
