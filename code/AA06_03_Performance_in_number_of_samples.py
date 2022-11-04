import numpy as np
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# read
pa1 = '/home/hulianting/Projects/Project08_ECG/data/result/'
two_step_res = np.load(pa1 + 'two_step_biglabel_metric.npy')
two_step_fewshot_res = np.load(pa1 + 'two_step_biglabel_few_shot_metric.npy')
m_ind = np.argmax(two_step_res[:, 6])
met08 = np.concatenate([two_step_res[m_ind, 0:2], np.array([0.8]), two_step_res[m_ind, 2:]])

two_step_fewshot_met = []
for i in range(1, 8):
    ind = np.where(two_step_fewshot_res[:, 2] == i/10)[0]
    res = two_step_fewshot_res[ind, :]
    ind = np.argmax(res[:, 7])


para = two_step_res[m_ind, :4].astype(np.int32)
para = str(para[0]) + str(para[1]) + str(para[2]) + str(para[3])
para_li = two_step_fewshot_res[:, [0, 1,3, 4]].astype(np.int32)
para_li = [str(para_li[x, 0]) + str(para_li[x, 1]) + str(para_li[x, 2]) + str(para_li[x, 3]) for x in range(para_li.shape[0])]
ind = np.where(np.array(para_li) == para)[0]
two_step_fewshot_met = two_step_fewshot_res[ind, :]

two_step_fewshot_met = np.concatenate([two_step_fewshot_met, met08[np.newaxis, :]], 0)
two_step_fewshot_met = two_step_fewshot_met[:, [2, 7] + list(range(16, 21))]

# read
pa2 = '/home/hulianting/Projects/Project08_ECG/data/Alternative models/0_super_classes/'
cnn08_res = sio.loadmat(pa2 + 'super_allclasses_cnn_mat.mat')['lstm_mat']
lstm08_res = sio.loadmat(pa2 + 'super_allclasses_lstm_mat.mat')['lstm_mat']

pa3 = '/home/hulianting/Projects/Project08_ECG/data/Alternative models/1_data_count/'
cnn_res, lstm_res = [], []
for i in range(1, 8):
    fi = pa3 + 'rate_' + str(i/10)[:3] + '_superclasses_cnn_mat.mat'
    res = sio.loadmat(fi)['cnn_mat']
    cnn_res.append(res)
    fi = pa3 + 'rate_' + str(i / 10)[:3] + '_superclasses_lstm_mat.mat'
    res = sio.loadmat(fi)['lstm_mat']
    lstm_res.append(res)

cnn_res.append(cnn08_res)
lstm_res.append(lstm08_res)

cnn_met, lstm_met = [], []
for i, v in enumerate(cnn_res):
    met = np.array([v[x, x] / np.sum(v[x, :]) for x in range(5)])
    met = np.concatenate([np.array([i/10]), np.array([np.mean(met)]), met])
    cnn_met.append(met)
cnn_met = np.stack(cnn_met, 0)

for i, v in enumerate(lstm_res):
    met = np.array([v[x, x] / np.sum(v[x, :]) for x in range(5)])
    met = np.concatenate([np.array([i/10]), np.array([np.mean(met)]), met])
    lstm_met.append(met)
lstm_met = np.stack(lstm_met, 0)


plt.figure(figsize=(12,9))
ti_li = ['macro', 'ventricular', 'atrial', 'atrioventricular junction', 'sinus', 'normal']
for i in range(6):
    ax = plt.subplot(2, 3, i + 1)
    ax.plot(two_step_fewshot_met[:, 0], two_step_fewshot_met[:, i+1])
    ax.plot(cnn_met[:, 0], cnn_met[:, i + 1])
    ax.plot(lstm_met[:, 0], lstm_met[:, i + 1])
    ax.title.set_text(ti_li[i])
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.tight_layout()
plt.show()
#plt.savefig('/home/hulianting/Projects/Project08_ECG/figures/compare/sample_num.svg')
plt.close()




