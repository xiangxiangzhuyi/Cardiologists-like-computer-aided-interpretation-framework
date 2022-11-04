import numpy as np
import wfdb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=80)
# file name
fi_na = '/home/hulianting/Projects/Project08_ECG/data/LUDB/data/2'
fi_li = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

# read data
record = wfdb.rdrecord(fi_na)
ann = [wfdb.rdann(fi_na, x) for x in fi_li]

for i in range(12):
    titl = fi_li[i]
    sing = record.p_signal[0: 5000, i]
    an_point = np.concatenate([[0], ann[i].sample, [5000]])
    an_symbol = [')'] + ann[i].symbol + ['(']

    # draw line
    figure(figsize=(30, 6), dpi=80)
    st, en, lab = 0, 0, 0
    for ii in range(len(an_point)):
        po, sy = an_point[ii], an_symbol[ii]
        if sy == ')':
            en = po
            plt.plot(np.arange(st, en), sing[st: en], color = lab)
            st = po
            lab = 'b'
        elif sy == 'N':
            lab = 'g'
            plt.plot(po, sing[po], 'o')
        elif sy == 't':
            lab = 'r'
            plt.plot(po, sing[po], 'v')
        elif sy == 'p':
            lab = 'c'
            plt.plot(po, sing[po], '1')
        elif sy == '(':
            en = po
            plt.plot(np.arange(st, en), sing[st: en], color=lab)
            st = po

    plt.show()
