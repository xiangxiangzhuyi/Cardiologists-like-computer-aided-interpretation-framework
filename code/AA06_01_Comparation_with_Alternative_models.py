import numpy as np
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# read
pa1 = '/home/hulianting/Projects/Project08_ECG/data/result/'
our_one_res = np.load(pa1 + 'one_step_metric.npy')
our_two_big_res = np.load(pa1 + 'two_step_biglabel_metric.npy')
our_two_fine_res = np.load(pa1 + 'two_step_finelabel_metric.npy')

# process our one step
our_one_res = our_one_res[np.where(our_one_res[:, 1] == 0)[0]]
ind = np.argmax(our_one_res[:, 7])
our_one_met = our_one_res[ind, [7] + list(range(16, 34))]

# process our two step
ind = np.argmax(our_two_big_res[:, 6])
our_two_big_met = our_two_big_res[ind, 15:20]
our_two_fine_met = []
fin_n = [4, 4, 5, 4]
for i in range(4):
    ind = np.where(our_two_fine_res[:, 1] == i)[0]
    fine_res = our_two_fine_res[ind, :]
    ind = np.argmax(fine_res[:, 7])
    met = fine_res[ind, 16:16+fin_n[i]]
    our_two_fine_met.append(met)

our_two_fine_met = [our_two_big_met[i]*v for i, v in enumerate(our_two_fine_met)]
our_two_fine_met.append(our_two_big_met[-1:])
our_two_fine_met = np.concatenate(our_two_fine_met)
order1 = [15,6,3,13,16,8,10,4,9,11,14,12,7,5,2,1,17,0]
our_two_met = our_two_fine_met[np.argsort(order1)]
our_two_met = np.concatenate([np.array([np.mean(our_two_met)]), our_two_met])

# read
pa2 = '/home/hulianting/Projects/Project08_ECG/data/other_files/'
cnn_res = sio.loadmat(pa2 + 'cnn_mat.mat')['cnn_mat']
lstm_res = sio.loadmat(pa2 + 'lstm_mat.mat')['lstm_mat']
order2 = [16,8,5,9,11,14,15,12,0,2,10,6,3,1,4,7,17,13]

# process cnn
cnn_met = np.array([cnn_res[x,x]/np.sum(cnn_res[x, :]) for x in range(18)])
cnn_met = cnn_met[np.argsort(order2)]
cnn_met = np.concatenate([np.array([np.mean(cnn_met)]), cnn_met])

# process lstm
lstm_met = np.array([lstm_res[x,x]/np.sum(lstm_res[x, :]) for x in range(18)])
lstm_met = lstm_met[np.argsort(order2)]
lstm_met = np.concatenate([np.array([np.mean(lstm_met)]), lstm_met])

# read
pa3 = '/home/hulianting/Projects/Project08_ECG/data/Alternative models/'
na_li = ['ada', 'KNN', 'RF', 'SVM', 'xgboost']
other_mat = []
for na in na_li:
    fi = pa3 + '2_' + na + '_results/' + na.lower() + '_mat.mat'
    mat = sio.loadmat(fi)[na.lower() + '_mat']
    other_mat.append(mat)

# process
other_met = []
for i, v in enumerate(other_mat):
    met = np.array([v[x, x] / np.sum(v[x, :]) for x in range(18)])
    met = met[np.argsort(order2)]
    met = np.concatenate([np.array([np.mean(met)]), met])
    other_met.append(met)

other_met = [our_one_met, our_two_met, cnn_met, lstm_met] + other_met

met_li = ['Macro', 'normal','ST','SB','IV block','AFL','SA','VT','PJC','AF','AV block','AT','JER','JE','PVC','JT','VE','PAC','SA block']
model = ['One-step', 'Two-step', '1D CNN', 'LSTM'] + ['Ada', 'KNN', 'RF', 'SVM', 'XGBoost']
da = []
for i, met in enumerate(met_li):
    for j, mod in enumerate(model):
        da.append([other_met[j][i], met, mod])
df = pd.DataFrame(da, columns = ['Recall', 'Arrhythmia category', 'Models'])


plt.figure(figsize=(21, 10))
sns.set(font_scale=1.1)
ax = sns.barplot(x="Arrhythmia category", y="Recall", hue="Models", data=df)
plt.show()
#plt.savefig('/home/hulianting/Projects/Project08_ECG/figures/compare/compare.svg')