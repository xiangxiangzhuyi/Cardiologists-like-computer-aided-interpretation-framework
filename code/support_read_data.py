import os
import numpy as np
import pickle
import random
import string

if os.path.isdir('/share/home/hulianting/Project/Project08_ECG/'):
    path = '/share/home/hulianting/Project/Project08_ECG/'
else:
    path = '/home/hulianting/Projects/Project08_ECG/'


def tr_te_da(sn, wn, ty_n, red_n, tre_ind):
    '''
    :param tr_r: the rate of training data
    :param sn: the sample size in each type
    :param wn: the number of waves spliced together
    :return: training data and test data
    '''
    file = path + 'data/split_pro_da/pro_da' + str(ty_n) + '.txt'
    with open(file, 'rb') as fi:
        clu_da = pickle.load(fi)
    # combine
    fea_arr = np.concatenate([np.stack(x, -1) for x in clu_da[1]], -1).astype(np.uint8)
    # convert
    ind_arr = np.stack(clu_da[0], 0)
    # divide the dataset
    all_sam = []
    for i in range(np.max(ind_arr[:, 0]) + 1):
        ind = np.where(ind_arr[:, 0] == i)[0]
        sam_n = np.max(ind_arr[ind, 1]) + 1
        all_sam.append([[] for x in range(sam_n)])

    # assign samples
    for i in range(ind_arr.shape[0]):
        if ind_arr[i, 0] != -1:
            all_sam[ind_arr[i, 0]][ind_arr[i, 1]].append(fea_arr[i, :])

    # divide the dataset
    tr_da, te_da = [], []
    for i in range(len(all_sam)):
        da = all_sam[i]
        [tr_ind, te_ind] = tre_ind[i]
        d_da = [da[x] for x in tr_ind]
        tr_da.append(d_da)
        d_da = [da[x] for x in te_ind]
        te_da.append(d_da)

    # remove unwanted classes
    con_lab = np.array([16,8,5,9,11,14,15,12,0,2,10,6,3,1,4,7,17,13])
    avi_ind = np.where(con_lab < 18 - red_n)[0]
    tr_da = [tr_da[x] for x in avi_ind]
    te_da = [te_da[x] for x in avi_ind]
    tre_ind = [tre_ind[x] for x in avi_ind]

    # training data
    tr_fea_li, tr_ind_li = [], []
    for i in range(len(tr_da)):
        num = int(np.ceil(sn/len(tr_da[i])))
        for j in range(len(tr_da[i])):
            sig = tr_da[i][j]
            if len(sig) == 0:
                continue
            com_ind = com_index(len(sig), wn)
            if num > com_ind.shape[0]:
                num = com_ind.shape[0]
            for k in range(num):
                sam = []
                for m in range(com_ind.shape[1]):
                    ind = com_ind[k, m]
                    sam.append(sig[ind])
                sam = np.concatenate(sam + [np.array([i]).astype(np.uint8)], 0)
                tr_fea_li.append(sam)
                tr_ind_li.append(tre_ind[i][0][j])

    tr_da = np.stack(tr_fea_li, 0)
    tr_ind = np.array(tr_ind_li).astype(np.int32)

    # test data
    te_li, te_ind_li = [], []
    for i in range(len(te_da)):
        for j in range(len(te_da[i])):
            sig = te_da[i][j]
            if len(sig) == 0:
                continue
            com_ind = com_index(len(sig), wn)
            if com_ind.shape[0] > 20:
                num = 20
            else:
                num = com_ind.shape[0]
            for k in range(num):
                sam = []
                for m in range(com_ind.shape[1]):
                    ind = com_ind[k, m]
                    sam.append(sig[ind])
                sam = np.concatenate(sam + [np.array([i]).astype(np.uint8)], 0)
                te_li.append(sam)
                te_ind_li.append(tre_ind[i][1][j])
    te_da = np.stack(te_li, 0)
    te_ind = np.array(te_ind_li).astype(np.int32)

    return tr_da, tr_ind, te_da, te_ind

# combien indexs
def com_index(sam_size, sam_num):
    ind_list = np.arange(sam_size, dtype=np.uint8)
    str_code = 'np.array(np.meshgrid('
    for j in range(sam_num):
        str_code = str_code + 'ind_list,'
    str_code = str_code + ')).T.reshape(-1, sam_num)'
    index = eval(str_code)

    so_ind = np.arange(index.shape[0])
    np.random.shuffle(so_ind)

    return index[so_ind]

# combine string
def com_mul_str(str_li):
    str_li = str_li + [''.join(random.sample(string.ascii_letters + string.digits, 4))]
    long_str = ''

    for s in str_li:
        long_str = long_str + '_' + str(s)

    return long_str[1:]





