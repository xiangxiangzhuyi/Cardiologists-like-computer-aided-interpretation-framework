import pickle
import numpy as np
import support_read_data as srd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics


sa_pa = '/home/hulianting/Projects/Project08_ECG/data/result/'
all_mat = np.load(sa_pa + 'one_step_metric.npy')
fi_na = '/home/hulianting/Projects/Project08_ECG/data/result/one_step'
with open(fi_na, 'rb') as fi:
    all_res = pickle.load(fi)

# draw figures
class_na = ['normal','ST','SB','IV block','AFL','SA','VT','PJC','AF','AV block','AT','JER','JE','PVC','JT','VE','PAC','SA block']

for rn in range(10):
    ind = np.where(all_mat[:, 1] == rn)[0]
    i_mat = all_mat[ind, 7]
    max_ind = ind[np.argmax(i_mat)]
    lab = all_res[max_ind][7]
    pred = all_res[max_ind][8]
    pr_lab = np.argmax(pred, -1)
    cm = confusion_matrix(lab, pr_lab)

    # convert
    con_lab = np.array([16, 8, 5, 9, 11, 14, 15, 12, 0, 2, 10, 6, 3, 1, 4, 7, 17, 13])
    avi_ind = np.where(con_lab < 18 - rn)[0]
    ind = np.argsort(con_lab[avi_ind])
    cm = cm[ind, :][:, ind]

    # draw CM
    '''
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1.6)
    ax = sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticklabels(class_na[:18-rn])
    ax.set_yticklabels(class_na[:18-rn])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.yticks(rotation=0)
    #plt.show()
    plt.savefig('/home/hulianting/Projects/Project08_ECG/figures/optimization/onestep_' + str(18 - rn) + ' classes_best_CM.svg')
    plt.close()
    '''
    # draw ROC
    plt.figure(figsize=(7, 7))
    for i in range(pred.shape[1]):
        ty = lab.copy()
        ty[ty != i] = -1
        ty[ty == i] = 1
        ty[ty == -1] = 0
        py = pred[:, i]
        fpr, tpr, _ = metrics.roc_curve(ty, py)
        AUC = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=class_na[i] + ' (AUC=' + str(AUC)[:5] + ')')
    plt.legend()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(str(18 - rn) + ' classes')
    #plt.show()
    plt.savefig('/home/hulianting/Projects/Project08_ECG/figures/optimization/onestep_' + str(18 - rn) + ' classes_ROC.svg')
    plt.close()




