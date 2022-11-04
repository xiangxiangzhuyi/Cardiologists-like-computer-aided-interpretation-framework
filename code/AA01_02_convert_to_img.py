import numpy as np
import wfdb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# save path
sa_pa = '/home/hulianting/Projects/Project08_ECG/data/LUDB/data_img/'
da_pa = '/home/hulianting/Projects/Project08_ECG/data/LUDB/data/'

for iid in range(1, 201):
    # file name
    fi_na = da_pa + str(iid)
    fi_li = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

    # read data
    record = wfdb.rdrecord(fi_na)
    ann = [wfdb.rdann(fi_na, x) for x in fi_li]

    fig, axs = plt.subplots(12, 1, figsize=(22, 25))

    for i in range(12):
        titl = fi_li[i]
        sing = record.p_signal[0: 5000, i]
        an_point = np.concatenate([[0], ann[i].sample, [5000]])
        an_symbol = [')'] + ann[i].symbol + ['(']

        # draw line
        st, en, lab = 0, 0, 0
        axs[i].set_title(titl)
        for ii in range(len(an_point)):
            po, sy = an_point[ii], an_symbol[ii]
            if sy == ')':
                en = po
                axs[i].plot(np.arange(st, en), sing[st: en], color = lab)
                st = po
                lab = 'b'
            elif sy == 'N':
                lab = 'g'
                axs[i].plot(po, sing[po], 'o')
            elif sy == 't':
                lab = 'r'
                axs[i].plot(po, sing[po], 'v')
            elif sy == 'p':
                lab = 'c'
                axs[i].plot(po, sing[po], '1')
            elif sy == '(':
                en = po
                axs[i].plot(np.arange(st, en), sing[st: en], color=lab)
                st = po

    plt.savefig(sa_pa + str(iid) + '.png')
    plt.close()