import support_based as spb
import support_read_data as srd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# read the dataset
fuer_da = srd.read_fuer_process('nor')
fu_pred = spb.read_result('ECG_seg_30000_50_3_all_18os', 'fu_pred_post_process.npy')
check_res = spb.read_result('ECG_seg_30000_50_3_all_18os', 'check.npy')

# traverse all data
ty31, ty11, ty12, ty22, ty23, ty33 = [], [], [], [], [], []
for i in check_res:
    st, en = 0, 0
    pri_li, ty_li = [], []
    for j in range(1, fu_pred.shape[1]):
        if fu_pred[i, j] != fu_pred[i, j-1]:
            en = j
            pri_li.append(fuer_da[i, st:en, :])
            ty_li.append(fu_pred[i, j-1])
            st = j

    pri_li.append(fuer_da[i, st:, :])
    ty_li.append(fu_pred[i, j-1])

    # remove the first and the last 500 points
    pri_len = [x.shape[0] for x in pri_li]
    for j in range(len(pri_len)):
        if np.sum(pri_len[:j]) > 500:
            st = j
            break
    for j in range(len(pri_len), 0, -1):
        if np.sum(pri_len[j:]) > 500:
            en = j
            break
    pri_li = pri_li[st:en]
    ty_li = ty_li[st:en]

    # add to the list
    for k in range(len(ty_li)):
        if pri_li[k].shape[0] < 5 or np.sum(np.abs(pri_li[k])) < 1:
            print(pri_li[k].shape[0], np.sum(np.abs(pri_li[k])))
            continue
        if ty_li[k] == 1:
            ty11.append(pri_li[k])
        elif ty_li[k] == 2:
            ty22.append(pri_li[k])
        elif ty_li[k] == 3:
            ty33.append(pri_li[k])
        elif ty_li[k] == 0 and k != 0 and k != len(ty_li)-1:
            if ty_li[k-1] == 3 and ty_li[k+1] == 1:
                ty31.append(pri_li[k])
            elif ty_li[k-1] == 1 and ty_li[k+1] == 2:
                ty12.append(pri_li[k])
            elif ty_li[k-1] == 2 and ty_li[k+1] == 3:
                ty23.append(pri_li[k])


# combine as a dist
primitive = {}
primitive['ty31'] = ty31
primitive['ty11'] = ty11
primitive['ty12'] = ty12
primitive['ty22'] = ty22
primitive['ty23'] = ty23
primitive['ty33'] = ty33


pa = '/home/hulianting/Projects/Project08_ECG/Model/model07_primitives/save_results/ECG_seg_30000_50_3_all_18os/'
with open(pa + 'primitive.txt', 'wb') as fi:
    pickle.dump(primitive, fi)











