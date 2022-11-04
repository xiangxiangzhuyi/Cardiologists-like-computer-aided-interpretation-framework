import pickle
import numpy as np
from sklearn.cluster import KMeans

# read the file
pa = '/home/hulianting/Projects/Project08_ECG/Model/model07_primitives/save_results/ECG_seg_30000_50_3_all_18os/'
with open(pa + 'points.txt', 'rb') as fi:
    point_dict = pickle.load(fi)

# convert to features
for ke in list(point_dict.keys()):
    poin_li = point_dict[ke]
    for i in range(len(poin_li)):
        fea = poin_li[i]
        fea = np.concatenate([fea[0]] + [fea[1][:, x] for x in range(12)])
        poin_li[i] = fea

    # normalize
    poin_li = np.stack(poin_li, 0)
    nor_poin_li = (poin_li - np.mean(poin_li, 0))/(np.std(poin_li, 0) + 0.0000000001)
    point_dict[ke] = nor_poin_li

# clustring
lab_dict = {}
for ke in list(point_dict.keys()):
    po_fea = point_dict[ke]
    kmeans = KMeans(n_clusters=50, random_state=0).fit(po_fea)
    lab_dict[ke] = kmeans.labels_

# save the result
pa = '/home/hulianting/Projects/Project08_ECG/Model/model07_primitives/save_results/ECG_seg_30000_50_3_all_18os/'
with open(pa + 'clustering_label.txt', 'wb') as fi:
    pickle.dump(lab_dict, fi)