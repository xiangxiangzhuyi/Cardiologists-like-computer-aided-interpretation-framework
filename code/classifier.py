import numpy as np
import datetime
import support_read_data as srd
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def Classification(sn, wn, ty_n, red_n, all_tr_te_ind, upda_t):

    # read data
    best_mean_acc, best_all_acc, best_te_l, best_te_p = 0, 0, 0, 0
    upt_res = []
    for ii in range(upda_t):
        tr_da, tr_index, te_da, te_index = srd.tr_te_da(sn=sn, wn=wn, ty_n=ty_n, red_n=red_n, tre_ind=all_tr_te_ind)
        # classification
        # training
        try:
            clf = CategoricalNB()
            clf.fit(tr_da[:, 0:-1], tr_da[:, -1])
            te_pred = clf.predict_proba(te_da[:, 0:-1])
            fl = 0
        except Exception:
            fl = 1
            pass

        if fl == 1:
            continue

        # removed index
        con_lab = np.array([16, 8, 5, 9, 11, 14, 15, 12, 0, 2, 10, 6, 3, 1, 4, 7, 17, 13])
        avi_ind = np.where(con_lab < 18 - red_n)[0]

        # combine
        te_l, te_p, te_ind = [], [], []
        for i in range(np.max(te_da[:, -1]) + 1):
            ind = np.where(te_da[:, -1] == i)[0]
            ty_te_l = te_index[ind]
            ty_te_p = te_pred[ind]
            for j in list(all_tr_te_ind[avi_ind[i]][1]):
                ind = np.where(ty_te_l == j)[0]
                p = np.mean(ty_te_p[ind, :], 0)
                te_l.append(i)
                te_p.append(p)
                te_ind.append(j)

        te_l = np.array(te_l)
        te_p = np.stack(te_p, 0)
        te_ind = np.array(te_ind)
        te_fl = np.argmax(te_p, -1)

        # metric
        all_acc = accuracy_score(te_l, te_fl)
        cm = confusion_matrix(te_l, te_fl)
        each_acc = np.array([cm[x, x] / np.sum(cm[x, :]) for x in range(cm.shape[0])])
        mean_acc = np.mean(each_acc)

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_all_acc = all_acc
            best_te_l = te_l
            best_te_p = te_p

        # appended to the list
        upt_res.append([mean_acc, all_acc, te_l, te_p, all_tr_te_ind])

        # update
        all_tr_te_ind = [[list(x[0]), list(x[1])] for x in all_tr_te_ind]

        # get error
        dif = te_l - np.argmax(te_p, -1)
        min_ind = np.where(dif != 0)[0]
        for i in range(min_ind.shape[0]):
            lab = te_l[min_ind[i]]
            lab = avi_ind[lab]
            ind = te_ind[min_ind[i]]
            all_tr_te_ind[lab][0].append(ind)
            all_tr_te_ind[lab][1].append(all_tr_te_ind[lab][0][0])
            all_tr_te_ind[lab][0].remove(all_tr_te_ind[lab][0][0])
            all_tr_te_ind[lab][1].remove(ind)

        all_tr_te_ind = [[np.array(x[0]), np.array(x[1])] for x in all_tr_te_ind]
        print(datetime.datetime.now(), ii, str(mean_acc)[:5], str(all_acc)[:5], str(best_mean_acc)[:5], str(best_all_acc)[:5])

    return best_mean_acc, best_all_acc, best_te_l, best_te_p, upt_res