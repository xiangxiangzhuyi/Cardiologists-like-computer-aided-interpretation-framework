import pickle
import classifier as cl
import support_read_data as srd
import os

def run_model(ty_n):
    # create folder
    sa_pa = srd.path + 'Model/model22_strong_and_weak/save_results/'
    fo = sa_pa + srd.com_mul_str([ty_n]) + '/'
    os.mkdir(fo)

    all_res = []
    sn = 50000000
    upda_t = 10
    for nor_num in [3, 4, 5]:
        sn = 50000000
        m_acc, a_acc, te_l, te_p, upt_res = cl.Classification(sn=sn, wn=wn, nor_num=nor_num, upda_t=upda_t)
        all_res.append([wn, red_n, sn, upda_t, ty_n, m_acc, a_acc, te_l, te_p, upt_res])

        # save the result
        with open(fo + 'res.txt', 'wb') as fi:
            pickle.dump(all_res, fi)
    return
