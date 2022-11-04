import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# read result
fi_na = '/home/hulianting/Projects/Project08_ECG/data/result/one_step'
with open(fi_na, 'rb') as fi:
    all_res = pickle.load(fi)

ty_str = ['macro', 'micro', 'weighted']
all_metric = []
for li in all_res:
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
    all_metric.append(li[:5] + mrt + [all_acc, mean_acc] + list(new_each_acc) + [0 for x in range(18-len(new_each_acc))])
all_metric = np.array(all_metric)

# save the result
sa_pa = '/home/hulianting/Projects/Project08_ECG/data/result/'
np.save(sa_pa + 'one_step_metric.npy', all_metric)

