import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set the path
sa_pa = '/home/hulianting/Projects/Project08_ECG/data/result/'
all_mat = np.load(sa_pa + 'one_step_metric.npy')

# draw
s_n_li = [3, 4, 5]
c_n_li = [3,5,8,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
all_cm = []
for cn in range(10):
    ind = np.where(all_mat[:, 1] == cn)[0]
    i_mat = all_mat[ind, :][:, [0, 4,7]]
    cm, cm_na = [], []
    for i in range(3):
        for j in range(18):
            i1 = np.where(i_mat[:, 0] == s_n_li[i])[0]
            i2 = np.where(i_mat[:, 1] == c_n_li[j])[0]
            i12 = np.intersect1d(i1, i2)
            cm.append(i_mat[i12, 2][0])
            cm_na.append(str(s_n_li[i]) + ',' + str(c_n_li[j]))
    all_cm.append(np.array(cm))

# draw
c_li = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][0]
plt.figure(figsize=(12,9))
for i in range(10):
    ax = plt.subplot(10, 1, i+1)

    # line
    x, y = np.arange(18*3), all_cm[i]
    ax.plot(x, y, color=c_li, alpha=0.8)
    ax.fill_between(x, y, facecolor=c_li, alpha=0.1)
    # maximum point
    print(np.argmax(y), np.max(y))
    ax.scatter(np.argmax(y), np.max(y))

    # set subplot
    ax.set_yticks([])
    ax.set_ylim(0.35, 1.05)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(-1, 18*3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color((0.1, 0.2, 0.5))
    ax.spines['bottom'].set_color((0.1, 0.2, 0.5))
# set the whole plot
ax.set_xticks(x)
ax.set_xticklabels(cm_na, rotation=90)

plt.show()
plt.close()
#plt.savefig('/home/hulianting/Projects/Project08_ECG/figures/optimization/one_step_opti.svg')






