
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from numpy import array
import numpy as np
from tqdm import trange
import os


def all_to_csv(dir_path='2024c/processed_data/'):
    for i in trange(len(os.listdir(dir_path))):
        fn = os.listdir(dir_path)[i]
        if os.path.isdir(dir_path+fn):
            continue
        df = read_csv(dir_path+fn, index_col=[0])
        if dir_path=='2024c/processed_data/':
            ori_df = read_csv('2024c/processed_data/origin/'+fn)
        else:
            ori_df = read_csv(dir_path+fn, index_col=[0])



        # 找特征: ：分差，赛点类别，连胜次数，是否投球
        score_dif_p1 = df['p1_points_won'] - df['p2_points_won']
        score_dif_p2 = df['p2_points_won'] - df['p1_points_won']
        # 标准化
        score_dif_p1 = (score_dif_p1-score_dif_p1.mean())/score_dif_p1.std()
        score_dif_p2 = (score_dif_p2-score_dif_p2.mean())/score_dif_p2.std()

        p1_break_points = df['p1_break_pt']
        p2_break_points = df['p2_break_pt']
        p1_break_points_won = df['p1_break_pt_won']
        p2_break_points_won = df['p2_break_pt_won']
        p1_ad = []
        for i in ori_df['p1_score']:
            if i == 'AD':
                p1_ad.append(1)
            else:
                p1_ad.append(0)
        p1_ad = array(p1_ad, dtype=np.int8)
        p2_ad = []
        for i in ori_df['p2_score']:
            if i == 'AD':
                p2_ad.append(1)
            else:
                p2_ad.append(0)
        p2_ad = array(p2_ad, dtype=np.int8)


        # 投球
        server = df['server']
        server_p1, server_p2 = [0]*len(server), [0]*len(server)
        for i,j in enumerate(server.tolist()):
            if j == 1: server_p1[i] = 1
            else: server_p2[i] = 1
        server_p1, server_p2 = array(server_p1), array(server_p2 )



        # 连胜
        p1_score = df['p1_score']
        p2_score = df['p2_score']

        from itertools import pairwise
        def comble_df(p_score):
            res = [0]
            cnt = 0
            for i,j in pairwise(p_score):
                if j > i:
                    cnt += 1
                if i == j or j == 0:
                    cnt = 0
                if cnt > 0:
                    res.append(cnt-1)
                else:
                    res.append(0)

            return array(res, dtype=np.int8)
        p1_comble = comble_df(p1_score)
        p2_comble = comble_df(p2_score)
        # 归一化
        p1_comble = (p1_comble-p1_comble.mean())/p1_comble.std()
        p2_comble = (p2_comble-p2_comble.mean())/p2_comble.std()


        # 标签: 接下来5场胜率
        def win_possiblity(window_size=5):
            point_victor = df['point_victor'].tolist()
            n = len(point_victor)
            point_victor += [1,1]+[0]*(window_size-2)
            res1, res2 = [], []
            for i in range(n):
                cnt1 = cnt2 = 0
                for j in point_victor[i:i+window_size]:
                    if j == 1:
                        cnt1 += 1
                    if j == 2:
                        cnt2 += 1
                res1.append(cnt1)
                res2.append(cnt2)
            return array(res1,dtype=np.int8), array(res2,dtype=np.int8)

        p1_win_possiblity, p2_win_possiblity = win_possiblity()


        n = len(score_dif_p1)
        for i in [score_dif_p1, score_dif_p2, server_p1, server_p2, p1_ad, p2_ad, p1_comble, p2_comble, p1_break_points, p2_break_points, p1_break_points_won, p2_break_points_won, p1_win_possiblity, p2_win_possiblity]:
            assert n == len(i)


        heatdf = DataFrame(
            dict(zip(['score_dif_p1','score_dif_p2','server_p1','server_p2','p1_ad','p2_ad','p1_comble','p2_comble','p1_break_points','p2_break_points','p1_break_points_won','p2_break_points_won','p1_win_possiblity','p2_win_possiblity'],
            [score_dif_p1, score_dif_p2, server_p1, server_p2, p1_ad, p2_ad, p1_comble, p2_comble, p1_break_points, p2_break_points, p1_break_points_won, p2_break_points_won, p1_win_possiblity, p2_win_possiblity]))
            )


        heatdf.head()
        if dir_path=='2024c/processed_data/':
            heatdf.to_csv('2024c/traintestds/'+fn)
        else:
            heatdf.to_csv('2024c/data/all_data_in_int.csv')
    print("all data to csv....")


if __name__ == '__main__':
    all_to_csv()
