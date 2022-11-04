import numpy as np
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# read
pa1 = '/home/hulianting/Projects/Project08_ECG/data/result/'
our_one_res = np.load(pa1 + 'one_step_metric.npy')
our_one_met = []
for i in range(10):
    ind = np.where(our_one_res[:, 1] == i)[0]
    res = our_one_res[ind, :]
    ind = np.argmax(res[:, 7])
    our_one_met.append(res[ind, 16:34-i])

# read
pa2 = '/home/hulianting/Projects/Project08_ECG/data/other_files/'
pa3 = '/home/hulianting/Projects/Project08_ECG/data/Alternative models/1_classes_count_trans/'
cnn18_res = sio.loadmat(pa2 + 'cnn_mat.mat')['cnn_mat']
lstm18_res = sio.loadmat(pa2 + 'lstm_mat.mat')['lstm_mat']
cnn_mat, lstm_mat = [],[]
for i in range(9, 18):
    mat = sio.loadmat(pa3 + str(i) +'classes_cnn_mat.mat')['cnn_mat']
    cnn_mat.append(mat)
    mat = sio.loadmat(pa3 + str(i) + 'classes_lstm_mat.mat')['lstm_mat']
    lstm_mat.append(mat)
cnn_mat.append(cnn18_res)
lstm_mat.append(lstm18_res)

# process
order = np.array([16,8,5,9,11,14,15,12,0,2,10,6,3,1,4,7,17,13])
cnn_met, lstm_met = [], []
for i, mat in enumerate(cnn_mat):
    met = np.array([mat[x, x] / np.sum(mat[x, :]) for x in range(mat.shape[0])])
    n_ord = np.array([x for x in order if x < mat.shape[0]])
    or_ind = np.argsort(n_ord[:mat.shape[0]])
    met = met[or_ind]
    cnn_met.append(met)

for i, mat in enumerate(lstm_mat):
    met = np.array([mat[x, x] / np.sum(mat[x, :]) for x in range(mat.shape[0])])
    n_ord = np.array([x for x in order if x < mat.shape[0]])
    or_ind = np.argsort(n_ord[:mat.shape[0]])
    met = met[or_ind]
    lstm_met.append(met)


# plt
met_li = ['normal','ST','SB','IV block','AFL','SA','VT','PJC','AF','AV block','AT','JER','JE','PVC','JT','VE','PAC','SA block']
our_one_met.reverse()
all_met = [our_one_met, cnn_met, lstm_met]
all_lines = []
for mod_met in all_met:
    lines = []
    for i in range(18):
        line = []
        for j, met in enumerate(mod_met):
            if len(met) > i:
                line.append([met.shape[0], met[i]])
        line = np.array(line)
        lines.append(line)
    all_lines.append(lines)


new_all_lines = [[all_lines[y][x] for y in range(3)] for x in  range(18)]
plt.figure(figsize=(12,9))
for i, lines in enumerate(new_all_lines):
    ax = plt.subplot(6, 3, i + 1)
    for j in range(3):
        line = lines[j]
        ax.plot(line[:, 0], line[:, 1])
    ax.title.set_text(met_li[i])
    ax.set_xticks(line[:, 0])
plt.tight_layout()
plt.show()
#plt.savefig('/home/hulianting/Projects/Project08_ECG/figures/compare/class_num.svg')
plt.close()

